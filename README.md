# Lab 2 — Relatório individual

Nome: Ivo Caetano de Andrade Junior  
Turma: Banco de Dados - 4o. Período
Link do repositório: https://github.com/icaj/classroom-lab3.git

## Reprodução
- **Python 3.13.2** (o enunciado recomenda 3.11/3.12; usei a 3.13 que já estava instalada e tudo rodou sem ajuste), numpy 2.2.3, pandas 2.2.3, scikit-learn 1.5.2, matplotlib 3.10.6, torch 2.14.1+cpu.
- Windows 11 Pro, CPU Intel Core i5-10210U (4 threads usadas pelo torch), sem GPU. Dispositivo: `cpu`.
- Semente: **42** (`fixar_semente(42)` antes de cada modelo; sementes 0–9 só na repetição da forma 4 da Parte C).
- Execução: `python -m pip install -r requirements.txt`, abrir `lab02.ipynb` e rodar *Restart & Run All*. Pela linha de comando: `jupyter nbconvert --to notebook --execute --inplace lab02.ipynb`. Leva menos de 1 minuto.
- `apoio.py`: **não modifiquei**.
- Tudo o que o notebook gera fica em `resultados/` (tabelas `.csv`, gráficos `.png`, pesos do modelo final e a decisão congelada em `d_decisao_congelada.json`).
- Os tempos da A3 mudam um pouco a cada execução; os valores deste relatório são da execução salva no notebook. Os demais números são determinísticos na mesma máquina.

## Base e definição do alvo
Red Wine Quality (Cortez et al., 2009), cópia da UCI incluída no pacote (CC BY 4.0, SHA-256 conferido por `carregar_base`). 1599 linhas, cada uma é uma amostra de vinho tinto com 11 medidas físico-químicas (as únicas entradas do modelo) e a nota sensorial `quality`. Alvo: `alvo = 1` se `quality >= 7` (217 linhas, 13,6%), senão 0. `quality`, `alvo`, `id_linha` e `grupo_perfil` nunca entram como característica (com exceção proposital e declarada do experimento errado da forma 2, na Parte C).

Limitações: há 240 linhas que repetem um perfil de medidas já existente (1359 perfis distintos); perfil igual não prova que é o mesmo vinho. Não há data, produtor, região nem custos, e a fronteira 7 é uma convenção do laboratório.

## B1 — Cinco cenários
1. **Avaliações de produtos.** Eu dividiria **por usuário** (GroupKFold/GroupShuffleSplit com o id do usuário), supondo que o modelo vai ser usado em avaliações de pessoas que ainda não estão na base. Numa divisão aleatória, textos do mesmo usuário (vocabulário, jeito de escrever, tendência a dar nota alta) aparecem no treino e na avaliação, e o modelo pode estar reconhecendo o autor em vez do sentimento. Se o uso for só para usuários antigos, a divisão aleatória (ou temporal por usuário) seria defensável. Também valeria checar se o mesmo produto aparece dos dois lados.
2. **Turbinas.** Divisão **temporal** (treinar no passado, avaliar num período posterior), de preferência também separando **turbinas inteiras** se a ideia for colocar o modelo em turbinas novas. Leituras de minuto em minuto são quase iguais às vizinhas, e o rótulo "falha nas próximas 24 h" é o mesmo para 1440 leituras seguidas; na divisão aleatória o modelo vê o minuto anterior e o seguinte da mesma janela de falha. Deixaria um intervalo de pelo menos 24 h entre treino e avaliação para o rótulo de um lado não usar informação do outro.
3. **Spam.** Aqui uma divisão **aleatória estratificada** (mantendo os 3% de spam em cada conjunto) pode ser defensável, se eu assumir que os e-mails são independentes e que o uso é no mesmo período. Os riscos que eu checaria antes: campanhas de spam quase idênticas (duplicatas que ficariam dos dois lados) e mudança ao longo do tempo, já que spammers mudam de estratégia. Se tiver data, prefiro avaliar num período posterior; e com 3% de positivos não usaria acurácia como métrica principal.
4. **Radiografias.** Divisão **por paciente** (todas as imagens de um paciente no mesmo conjunto), supondo uso em pacientes novos. São em média ~9 imagens por paciente e elas carregam anatomia, aparelhos, marcações e até o mesmo achado; numa divisão aleatória o modelo pode reconhecer o paciente. Se eu soubesse o hospital ou o equipamento, consideraria também separar por origem para testar outro serviço, mas essa informação não foi dada.
5. **Transações de cartão, tempo e clientes.** O principal é o **tempo**: treinar em 2019–2024, validar em 2025 e testar em 2026 (por exemplo), porque fraude muda e o modelo vai ser usado no futuro. Na divisão aleatória entram informações do futuro (padrões de golpes posteriores, agregados do cliente calculados com transações que vêm depois). Sobre **identidade**: se o objetivo é atender **clientes conhecidos**, é aceitável que o mesmo cliente apareça no treino (passado) e no teste (futuro), desde que o histórico usado seja só o anterior à transação. Se o objetivo é **clientes novos**, preciso separar também por cliente (clientes do teste nunca vistos no treino), senão o modelo aproveita o "jeito" de cada cliente e a métrica fica otimista. Nesse caso eu reportaria as duas avaliações separadas.

## B2 — Auditoria da separação
**Pergunta de uso e estratégia:** quero saber como o modelo classifica perfis de medidas que ele nunca viu. Usei `GroupShuffleSplit` com `grupo_perfil` como grupo e semente 42: primeiro 20% dos grupos para o teste, depois 25% dos grupos restantes para a validação. Não escolhi a semente por métrica nem por proporção de classe; só conferi depois.

| Conjunto | Linhas | Positivos | Proporção positiva | Perfis |
|---|---:|---:|---:|---:|
| Treino | 961 (60,1%) | 139 | 14,46% | 815 |
| Validação | 318 (19,9%) | 33 | 10,38% | 272 |
| Teste | 320 (20,0%) | 45 | 14,06% | 272 |

- Cobertura: 961 + 318 + 320 = 1599, todas as linhas presentes exatamente uma vez.
- Linhas em comum: treino×validação 0, treino×teste 0, validação×teste 0.
- Perfis (11 medidas) em comum: 0 nos três pares.
- Ambas as classes nos três conjuntos (a validação ficou com menos positivos, 33; isso deixa a escolha de limiar mais instável).
- Para mostrar que índice disjunto não basta: numa divisão aleatória simples por linha, 64 linhas do teste teriam um gêmeo exato no desenvolvimento.
- O que essa partição **não** testa: futuro temporal, novos produtores, outras regiões ou vinho branco. A base não tem esses metadados.

## A1 — Dataset
`MeuDataset` recebe o DataFrame de uma partição, a lista de 11 características e, opcionalmente, média e desvio. No treino calcula média e desvio **populacional (ddof=0)**; na validação e no teste recebe os vetores do treino (conferido com `np.array_equal`). Desvio zero seria trocado por 1 (não ocorreu no treino). Comprimentos 961 / 318 / 320; um exemplo devolve X `[11]` float32 e y escalar `[]` float32; um lote do treino devolve X `[64, 11]` e y `[64]`, ambos float32. Nenhum NaN ou infinito. A média normalizada da validação/teste não é zero (vai até ~0,13 em algumas colunas), o que é esperado.

## A2 — DataLoader
| Conjunto | batch_size | shuffle | drop_last | num_workers | Justificativas |
|---|---|---|---|---|---|
| Treino | 64 | True | False | 0 | 64 dá 16 passos por época (15 cheios + 1 com 1 exemplo), equilíbrio entre ruído e nº de atualizações. Embaralhar muda a composição dos lotes a cada época (não muda quais linhas são de treino). Não descarto porque a rede principal não tem BatchNorm; o custo é um passo ruidoso com 1 exemplo por época. 0 workers: dados já em memória e base pequena (~7 ms por passada); com 2 workers os processos falharam no Windows (spawn não importa a classe do notebook). |
| Validação | 256 | False | False | 0 | Sem gradiente, o tamanho só afeta velocidade (2 lotes). Sem embaralhar e sem descartar: todas as 318 linhas avaliadas, na mesma ordem dos rótulos (conferido). Workers: mesmo motivo do treino. |
| Teste | 256 | False | False | 0 | Mesma lógica da validação: 320 linhas, ordem fixa, nada descartado. |

## A3 — Tempo por época
Treino completo (961 exemplos), rede sem BatchNorm, mesmo estado inicial e Adam novo a cada medição, 1 época de aquecimento descartada, `time.perf_counter()`, CPU i5-10210U.

| Lote | Repetição 1 (s) | Repetição 2 (s) | Repetição 3 (s) | Mediana (s) | Passos | ms/passo |
|---|---|---|---|---|---|---|
| 1 | 0,8882 | 0,8839 | 0,8968 | 0,8882 | 961 | 0,92 |
| 8 | 0,1613 | 0,1652 | 0,1646 | 0,1646 | 121 | 1,36 |
| 64 | 0,0307 | 0,0291 | 0,0340 | 0,0307 | 16 | 1,92 |
| 512 | 0,0129 | 0,0126 | 0,0128 | 0,0128 | 2 | 6,38 |

Gráfico: `resultados/a3_tempo_por_lote.png`. De 1→8 e de 8→64 a época ficou ~5x mais rápida a cada salto; de 64→512 só ~2,4x, então o ganho diminui a partir de 64. O custo de cada passo cresce com o lote, mas bem menos que proporcionalmente: com lotes pequenos domina o custo fixo de cada passo (Python, zerar gradiente, Adam) e com lote grande a conta começa a pesar. Lotes diferentes também significam número diferente de atualizações (961 contra 2 por época), então época mais rápida não quer dizer convergência mais rápida; isso eu não medi.

## A4 — BatchNorm
Subconjunto de 65 linhas **do treino**, lote 64, resto 1 (conferido com `assert`). Com `drop_last=False` os lotes são [64, 1] e a rede com `BatchNorm1d` em `train()` falha no segundo:

`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 16])`

Correção: `drop_last=True` no carregador de treino, mantendo `train()` e a BatchNorm usando estatísticas do lote. Custo: 1 de 65 observações (≈1,5%) fica fora a cada época; com embaralhamento quem sai muda, e em 20 épocas as 65 linhas foram usadas. Para um resto maior eu preferiria trocar o tamanho do lote ou usar LayerNorm. Na avaliação nada é descartado.

## C — Duas formas de vazamento
Desenvolvimento interno `c_dev` = treino + validação da B2 (1279 linhas), dividido por grupo 75/25 (semente 42) em `c_treino` (961) e `c_aval` (318). Modelo: `RandomForestClassifier(n_estimators=200, random_state=42)`. Métrica: acurácia (AUC-ROC como apoio).

| Forma | Dados e partições internas | Métrica | Incorreto | Corrigido | Diferença/unidade |
|---|---|---|---|---|---|
| 2 — alvo entre as características | Wine real; `c_treino`/`c_aval` por grupo | Acurácia em `c_aval` | 1,0000 (AUC 1,000) | 0,8994 (AUC 0,805) | +10,06 p.p. |
| 4 — duplicatas entre treino e avaliação | Wine real; `c_dev` 75/25, por linha (incorreto) vs por grupo (corrigido), sementes 0–9 | Acurácia média (10 sementes) | 0,9084 | 0,8791 | +2,93 p.p. (desvio 1,93) |

**Experimento 1 (forma 2).** A versão errada usa as 11 medidas **mais a coluna `quality` original, sem transformação**, como 12ª entrada. A nota só existe depois da prova sensorial e é a origem do alvo. A floresta acerta tudo e dá importância 0,69 para `quality`. A diferença em acurácia parece pequena (10 p.p.) porque 89,6% de `c_aval` é classe 0 e a versão correta já está em 89,9%; o vazamento pegou todo o espaço que sobrava. A AUC (0,80 → 1,00) mostra melhor o tamanho do problema.

**Experimento 2 (forma 4).** A versão errada divide `c_dev` por linha, e em média 69 linhas da avaliação (≈22%) têm gêmeo exato no treino. Nessas linhas o modelo acertou 100% em todas as sementes; nas outras, 88,3%, praticamente igual à divisão correta. O ganho vem de memória. O incorreto ficou acima em 9 de 10 sementes; a exceção (semente 6) teve uma avaliação por grupo com só 9,8% de positivos. Como a correção muda a partição, os conjuntos avaliados não são idênticos; por isso usei 10 sementes fixas antes de rodar e reporto média e desvio, sem descartar nenhuma.

**Como preservei o teste principal:** a Parte C só usa `c_dev`, há um `assert` garantindo que nenhum `id_linha` do teste está nele, todas as variáveis têm prefixo `c_` e o modelo da C não é usado na D. Obs.: com a mesma semente e proporção, `c_treino`/`c_aval` coincidiram com treino/validação da B2; isso não afeta a conclusão porque tudo continua dentro do desenvolvimento.

## D7 — Treinamento e parada
Rede de apoio 11 → 16 → ReLU → 1 (sem BatchNorm), Adam lr=0,001, `BCEWithLogitsLoss` sem peso, lote 64. Máximo 100 épocas, paciência 10, melhora mínima 0,0001 na **perda de validação**.

**A paciência disparou:** melhor época **49** (perda de validação 0,26523), parada na época **59**. Pesos da época 49 restaurados a partir de uma cópia com `copy.deepcopy`; conferi recalculando a perda de validação do modelo restaurado (0,26523, contra 0,26574 da última época). O loader de teste não é passado para a função de treino.

Curvas: `resultados/d7_curvas_perda.png`. A validação fica levemente abaixo do treino (a perda de treino é média durante a época e a validação tem menos positivos); depois da época ~30 as curvas quase não mudam.

## D4 — Matriz de custos
Prever positivo = mandar a amostra para a análise especial; negativo = não analisar.

| | Previsto 0 | Previsto 1 |
|---|---|---|
| Real 0 | 0 | C_FP = 1 |
| Real 1 | C_FN = 3 | 0 |

Unidade: "unidades de esforço". FP = uma análise desperdiçada em vinho com nota < 7. FN = um vinho de nota >= 7 que não foi selecionado. Considerei o FN 3x pior porque achar esses vinhos (raros, ~14%) é o propósito da análise.

**Esses custos são hipóteses do exercício.** A base não tem preços, custos de análise ou valor dos vinhos; o que importa é a razão 1:3 e ela deveria ser revista com dados reais.

## D5 e D6 — Escolha na validação
- Candidatos: todos os 272 escores únicos da validação + 0 (tudo positivo) + infinito (tudo negativo), 274 no total, sem grade. Positivo se `escore >= limiar`. Desempate: maior limiar de custo mínimo (não houve empate).

| Regra (validação, 318 obs.) | Limiar | FP | FN | Selecionadas | Custo total | Custo/obs. |
|---|---|---|---|---|---|---|
| Padrão | 0,5 | 12 | 25 | 20 | 87 | 0,274 |
| Escolhido | 0,2766 | 33 | 9 | 57 | 60 | 0,189 |
| Fórmula | 0,25 | 35 | 9 | 59 | 62 | 0,195 |

- A fórmula C_FP/(C_FP+C_FN) = 0,25 ficou muito perto do limiar da busca (diferença de 2 FP na validação). Ainda assim não trato isso como prova de calibração: Brier 0,079 (contra 0,095 prevendo sempre a prevalência do treino), escore médio 0,127 contra 10,4% de positivos na validação, e o bin mais alto do diagrama tem escore médio 0,49 com 37,5% de positivos. Com 33 positivos, tanto o diagrama quanto o ponto mínimo são ruidosos. Não usei pesos de classe nem reamostragem.
- A mesma validação serviu para a parada e para o limiar, então os números acima são de desenvolvimento (otimistas).
- **Configuração congelada antes do teste:** modelo da época 49, 11 entradas, média/desvio do treino, C_FP=1, C_FN=3, limiar 0,2766, regra `>=` (salvos em `resultados/d_decisao_congelada.json` e `d_modelo_final.pt`).

## D1 e D2 — Avaliação final
Teste: 320 observações, 45 positivas (**14,06%**). Classe majoritária no treino: 0. Convenção: sem previsões positivas, precisão e F1 = 0 (`zero_division=0`). AUCs calculadas com escores. AUC-PR = área trapezoidal (`auc(recall, precision)`); AP reportada separada.

Matrizes (linhas = real, colunas = previsto, ordem [0, 1]):

| Limiar 0,5 | prev 0 | prev 1 |
|---|---|---|
| real 0 | 269 | 6 |
| real 1 | 33 | 12 |

| Limiar 0,2766 | prev 0 | prev 1 |
|---|---|---|
| real 0 | 246 | 29 |
| real 1 | 17 | 28 |

| Regra | Limiar | Acurácia | Precisão | Revocação | F1 | AUC-ROC | AUC-PR (trap.) | AP | Custo/observação |
|---|---|---|---|---|---|---|---|---|---|
| Trivial do treino (sempre 0) | não se aplica | 0,8594 | 0 (conv.) | 0,0000 | 0 (conv.) | não reportada | não reportada | não reportada | 0,4219 |
| Rede: padrão | 0,5 | 0,8781 | 0,6667 | 0,2667 | 0,3810 | 0,8906 | 0,5641 | 0,5736 | 0,3281 |
| Rede: escolhido na validação | 0,2766 | 0,8562 | 0,4912 | 0,6222 | 0,5490 | 0,8906 | 0,5641 | 0,5736 | 0,2500 |

O trivial tem 85,9% de acurácia sem achar nenhum vinho bom. A rede com 0,5 ganha só 1,9 p.p. de acurácia e acha 12 de 45. O limiar escolhido tem a **menor** acurácia das três, mas acha 28 de 45 e tem o menor custo (80 contra 105 e 135). Para o trivial as AUCs não foram reportadas: um escore constante não forma curva. As AUCs da rede são iguais nas duas linhas porque o limiar não muda o ordenamento. O custo no teste (0,25/obs) ficou acima do visto na validação (0,19/obs), como esperado.

Arquivos: `resultados/d2_tabela_final_teste.csv`, `resultados/d2_matrizes_confusao.png`.

## D3 — Curvas
`resultados/d3_roc_pr_teste.png`: ROC (AUC-ROC 0,891) com a diagonal aleatória de (0,0) a (1,1); PR (AUC-PR trapezoidal 0,564; AP 0,574) com a linha horizontal na prevalência do teste (0,141).

## D8 — Recomendação
O modelo pode apoiar a **triagem** de quais amostras de vinho tinto mandar para a análise especial de alta nota (quality >= 7), a partir das 11 medidas físico-químicas. Recomendo o limiar **0,2766**, fixado na validação com custos **hipotéticos** de 1 unidade por análise desperdiçada (FP) e 3 por vinho bom não selecionado (FN). No teste reservado (320 amostras, 14,1% positivas), essa regra encontrou 28 dos 45 vinhos bons (revocação 62%), com precisão de 49% (cerca de metade das amostras enviadas são realmente de alta nota) e custo de 0,25 unidade por amostra, contra 0,33 no limiar 0,5 e 0,42 sem modelo nenhum. O erro que mais merece atenção ainda é o falso negativo: 17 vinhos bons (38%) passariam despercebidos; se a análise for barata, vale baixar o limiar e aceitar mais envios. Limitações: os custos são suposições e não números da operação; o resultado vale para esta base (vinhos tintos portugueses do estudo de Cortez et al.), sem datas, produtores ou regiões, então não garante o mesmo desempenho em outras safras, outros produtores ou vinho branco; a validação tem poucos positivos (33), o que deixa o limiar instável; e os escores da rede não estão bem calibrados no topo, então não devem ser lidos como probabilidade de o vinho ser bom.

## Referências e limites
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T (CC BY 4.0).
- Documentação: PyTorch Dataset/DataLoader, BatchNorm1d e BCEWithLogitsLoss; scikit-learn GroupShuffleSplit, métricas de classificação, `calibration_curve` e o guia "Common pitfalls".
- Incertezas: conjuntos pequenos (33 positivos na validação, 45 no teste), então as métricas têm variação considerável; não calculei intervalos de confiança. Os custos são hipotéticos. Os tempos da A3 dependem da máquina e da carga no momento.
- Não avaliado: generalização temporal, para novos produtores/regiões, para vinho branco e convergência em função do tamanho do lote.
