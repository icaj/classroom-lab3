# Laboratório 2 — Ciclo de treino, vazamento e métricas

**Deep Learning · CESAR School · Entrega individual · Valor: 1,0 ponto na AV1**

**Prazo: 08/10/2026.** O horário limite é o cadastrado na atividade do Classroom.

Neste laboratório, a pergunta é: **que evidências permitem confiar no resultado de um classificador?** Você construirá uma forma de carregar dados, justificará sua separação, reproduzirá dois vazamentos e produzirá um relatório de métricas com uma decisão de limiar. A avaliação considera a execução e a explicação do processo; não exige atingir uma acurácia mínima.

Esta versão usa uma base pública comum à turma. Você **não precisa ter um banco do projeto pronto**. As Partes A, B2 e D usam Red Wine Quality; a Parte C usa uma cópia de desenvolvimento dessa base ou outro conjunto claramente identificado. Os cinco cenários da B1 são exercícios de raciocínio e não exigem baixar outras bases.

## 1. Comece por aqui

1. Extraia o ZIP inteiro. Não abra o notebook dentro do arquivo compactado.
2. Leia o problema e o dicionário em `dados/DICIONARIO.md`.
3. Abra `lab02.ipynb`. Execute a preparação e confira a leitura do CSV.
4. Faça B2 antes de A: os conjuntos precisam estar separados antes de calcular média e desvio.
5. Complete A, C e D. A ordem de programação da D começa pela parada antecipada (D7), depois pela escolha de custos e limiar, e termina na avaliação final.
6. Preencha `README.md` e `USO-DE-IA.md`. Reinicie o ambiente e execute tudo antes de enviar.

O notebook é um ponto de partida: contém explicações, funções de apoio e lugares marcados **TODO** para as implementações avaliadas. Mensagens de “pendente” mostram o que falta; elas não contam como resultado. Os exemplos pequenos de preparação apenas verificam o ambiente.

### Rodar no computador

Use Python 3.11 ou 3.12. Abra um terminal dentro da pasta extraída:

```bash
python -m venv .venv
```

No Windows, ative com `.venv\Scripts\activate`; no Linux/macOS, com `source .venv/bin/activate`. Depois:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Selecione o ambiente instalado como kernel. Não é necessário ter GPU. Se houver problema com processos auxiliares do carregamento, comece com `num_workers=0` e justifique essa escolha.

### Rodar no Google Colab

Abra um notebook novo, envie o ZIP pela aba de arquivos e execute:

```python
from pathlib import Path
import zipfile

pacote = Path('/content/classroom-lab-02.zip')
with zipfile.ZipFile(pacote) as arquivo:
    arquivo.extractall('/content')
```

Em **Arquivo → Fazer upload de notebook**, abra `lab02.ipynb` que está no pacote. No início desse notebook, use `%cd /content/classroom-lab-02`. O módulo `apoio.py` e a pasta `dados/` devem estar no mesmo diretório de trabalho. O Colab normalmente já contém as bibliotecas principais; execute a célula de versões. Se faltar uma biblioteca, instale-a no próprio ambiente. Ao terminar, baixe o notebook executado: os arquivos do ambiente temporário não ficam automaticamente salvos no seu computador.

## 2. A base comum e a pergunta de classificação

A base sugerida é [Red Wine Quality no Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). O CSV deste pacote vem da [fonte original UCI](https://archive.ics.uci.edu/dataset/186/wine+quality), com autoria e licença registradas em `dados/FONTE-E-LICENCA.md`. Assim, a atividade não depende de login no Kaggle para começar.

Uma linha representa uma observação de vinho tinto. Há **11 medidas numéricas** e uma nota sensorial chamada `quality`. Para este exercício, defina:

```python
y = (df['quality'] >= 7).astype('int64')
```

**Classe 1:** nota pelo menos 7. **Classe 0:** nota abaixo de 7. Esta fronteira é uma convenção do laboratório, não uma regra universal de qualidade. Imagine um sistema que seleciona amostras para uma análise especial: interessa encontrar a classe 1 usando apenas as medidas disponíveis antes da nota sensorial.

`quality` dá origem à resposta. Portanto, **não pode entrar nas características do modelo correto**. Também não entram `alvo`, identificadores de linha nem códigos criados para controlar a partição. Use exclusivamente as 11 colunas listadas no dicionário.

O arquivo é pequeno, tabular, real e desbalanceado depois da transformação do alvo. Não exige tratamento de texto ou imagens. Consulte as contagens verificadas no dicionário. Não confunda desbalanceamento moderado com o caso extremo de fraude discutido na aula.

### Uma particularidade importante: perfis repetidos

Existem linhas com as mesmas 11 medidas. O pacote preserva o arquivo original. Para a avaliação principal, mantenha **perfis idênticos no mesmo conjunto**. Você pode agrupar pelas 11 características, sem usar o alvo, e fazer a separação por esses grupos. O notebook calcula `grupo_perfil` apenas para facilitar a identificação; esse código não é entrada do modelo.

Dois perfis iguais não provam que o vinho físico é o mesmo: a base não contém um identificador confiável dessa entidade. A regra é uma precaução didática para não avaliar repetidamente uma configuração que já apareceu no treino. Registre essa limitação. Também não há datas de coleta para fundamentar uma avaliação temporal real; não invente datas ou produtores para esta base.

## 3. O que entregar e em que ordem trabalhar

No repositório pessoal, organize:

```text
lab-02/
  README.md             respostas, tabelas, gráficos e conclusão
  lab02.ipynb           notebook executado, saídas visíveis
  USO-DE-IA.md          declaração obrigatória
  apoio.py             apoio utilizado; registre se modificar
  requirements.txt     dependências
  dados/               CSV, dicionário e atribuição
  resultados/          imagens e tabelas que seu notebook exportar
```

Envie **o link do repositório/pasta no Classroom**, conforme o canal de entrega da turma. O ZIP disponibilizado pelo professor é o material de partida, não a sua entrega pronta. Não envie a pasta `.venv` nem caches de execução.

| Parte | Valor | Evidência principal |
|---|---|---|
| A — Dataset e DataLoader | 0,20 | classe própria, justificativas, tempos e erro de BatchNorm |
| B — Partição | 0,20 | cinco análises e auditoria numérica da separação |
| C — Vazamento | 0,30 | dois experimentos com comparação e explicação |
| D — Métricas e decisão | 0,30 | modelo com parada antecipada, limiar escolhido na validação e relatório final |

**Fluxo recomendado:** leitura → B2 → A → reserva do teste → C isolada no desenvolvimento → D7 → D4–D6 → D1–D3 → D8 → revisão da entrega.

## 4. Parte A — Dataset e DataLoader · 0,20

### A1. Implemente seu Dataset

Crie uma classe que herde de `torch.utils.data.Dataset`. Receba uma tabela de uma única partição, a lista de características e, opcionalmente, média e desvio já calculados. Escreva docstrings para explicar `__init__`, `__len__` e `__getitem__`.

No treino, aprenda as estatísticas apenas com suas linhas. Na validação e no teste, receba e reutilize **os mesmos valores do treino**. Trate desvio zero para não produzir divisão por zero. Declare a convenção de desvio usada; pequenas diferenças entre implementações podem vir do denominador da variância.

Para um exemplo, devolva `X` com forma `[11]`, tipo `float32`, e `y` escalar `float32`. Em um lote de tamanho B, as formas esperadas são `[B, 11]` e `[B]`. A rede de apoio devolve B logits; `BCEWithLogitsLoss` recebe logits diretamente, sem aplicar sigmoide antes da perda.

**Mostre:** comprimento do Dataset, formas e tipos de um exemplo e de um lote; ausência de `NaN`/infinito; e evidência de que validação e teste usaram as estatísticas do treino. Não exija média zero da validação: isso não é condição de correção.

### A2. Configure três carregadores

Crie `loader_treino`, `loader_validacao` e `loader_teste`. Para cada um, justifique `batch_size`, `shuffle`, `drop_last` e `num_workers`. Diga qual é o efeito da escolha naquele conjunto; “é o padrão” não é justificativa.

Preserve todas as observações da avaliação. A ordem das previsões deve continuar alinhada à dos rótulos. Embaralhar o treino não troca a separação das partições. Múltiplos workers podem custar mais que ajudam numa base pequena; meça antes de alegar que aceleram.

**Entregue:** tabela com os três carregadores e seus quatro parâmetros, acompanhada das justificativas.

### A3. Meça o tempo de uma época

Compare `batch_size` **1, 8, 64 e 512**. Uma época completa percorre o conjunto de treino e executa previsão, perda, retropropagação e atualização. Medir só a criação do DataLoader não atende ao item.

Use a mesma rede, dados, dispositivo e configuração do otimizador. Comece cada medição do mesmo estado inicial e com um otimizador novo; não continue treinando o modelo da medição anterior. Use uma rede **sem BatchNorm neste item**, para que o lote 1 não interrompa a comparação. Não descarte observações para favorecer determinado tamanho de lote.

Faça um aquecimento fora da medição e, de preferência, três repetições por tamanho. Registre a mediana, o número de exemplos e de passos. Na GPU, sincronize antes e depois do intervalo; na CPU, `time.perf_counter()` basta. Salve o gráfico tempo × tamanho do lote, com unidades e descrição do equipamento.

**Explique:** onde o ganho diminuiu, o custo por passo e por que lotes diferentes também produzem quantidades diferentes de atualizações. Não invente um ponto de retorno decrescente se sua medição não o mostrou.

### A4. Reproduza e corrija o erro de BatchNorm

Separe um subconjunto **apenas do treino** cujo tamanho deixe resto 1 na divisão pelo tamanho do lote. Por exemplo, 65 observações e lote 64 satisfazem a condição; confirme-a numericamente. Use uma rede com `nn.BatchNorm1d` e modo `train()`.

Execute com `drop_last=False`, capture o erro esperado e imprima sua mensagem. Depois implemente uma correção apropriada e explique o custo da escolha: há observações descartadas, uma mudança de normalização ou outra alteração? A captura deve ser específica ao experimento; não esconda todos os erros do notebook com um `except` genérico.

**Entregue:** tamanho e resto, tamanho do lote que falhou, mensagem original, correção e justificativa. Esse subconjunto serve só à demonstração; não substitui a base principal. Mudar toda a rede para avaliação durante o treinamento não demonstra uma correção adequada do procedimento.

## 5. Parte B — A partição correta · 0,20

### B1. Analise os cinco cenários

Para cada linha, escreva de três a cinco linhas: qual partição adotaria, qual uso futuro está pressupondo e que informação poderia atravessar os conjuntos numa divisão aleatória. Quando faltarem informações, declare a hipótese em vez de fabricar certeza. Em cenários sem dependência relevante, a divisão aleatória pode ser defensável; justifique.

| Nº | Cenário |
|---|---|
| 1 | 50.000 avaliações de produtos escritas por 3.000 usuários. Prever se a avaliação é positiva. |
| 2 | Leituras de vibração de 20 turbinas, de minuto em minuto, ao longo de dois anos. Prever falha nas próximas 24 horas. |
| 3 | 200.000 e-mails rotulados como spam ou não, com 3% de spam. |
| 4 | 8.000 radiografias de tórax de 900 pacientes. Prever presença de achado. |
| 5 | Transações de cartão de 2019 a 2026, de 500 mil clientes. Prever fraude. |

No cenário 5, discuta explicitamente **tempo e identidade do cliente**. Diga como o objetivo de atender clientes conhecidos ou novos altera a avaliação. Não é necessário implementar esses cinco bancos nem escolher o mesmo tipo de partição para todos.

### B2. Implemente e audite a base comum

Para Wine Quality, declare a pergunta de uso e implemente treino, validação e teste com proporções **aproximadas de 60%, 20% e 20%**. Use semente fixa e mantenha perfis idênticos juntos. Uma API possível é `GroupShuffleSplit`, aplicada primeiro para reservar o teste e depois para dividir o desenvolvimento. As proporções dessa API se referem a grupos; as quantidades de linhas podem variar.

Sua auditoria deve mostrar: quantidade de linhas por conjunto; identificadores de linha sem sobreposição; cobertura de todas as linhas; número de perfis compartilhados entre cada par de conjuntos igual a zero; presença de ambas as classes e proporções observadas. Não procure uma semente pela métrica final. Use a conferência de classes apenas como validação do desenho, antes de modelar.

**Entregue:** tabela de partições e verificações numéricas, acompanhadas de uma justificativa. Ausência de sobreposição de índices, sozinha, não prova ausência de repetição de características. Explique também o que a base não permite testar: futuro temporal, novos produtores e outros domínios não estão documentados.

## 6. Parte C — Vazamento reproduzido e quantificado · 0,30

Escolha **duas formas diferentes** da lista. Pelo menos uma deve usar dados reais; a base incluída atende a isso. Pelo menos uma deve ser das formas **2, 4 ou 7**, como no enunciado original.

| Forma | Mecanismo que deve ser demonstrado |
|---|---|
| 1 | Aprender o pré-processamento com dados que deveriam estar reservados. |
| 2 | Incluir o alvo ou uma informação derivada dele entre as características. |
| 3 | Usar informação que só existirá depois do momento da previsão. |
| 4 | Permitir duplicatas entre os conjuntos de aprendizagem e avaliação. |
| 5 | Misturar a mesma entidade em conjuntos que deveriam representar entidades novas. |
| 6 | Selecionar características usando também as respostas da avaliação. |
| 7 | Usar repetidamente o teste para escolher modelo, configuração ou limiar. |

Um caminho simples é investigar **1 e 2 na base comum**. Outra possibilidade é estudar perfis repetidos na forma 4. A escolha e a implementação são suas. Para os mecanismos temporal e por entidade real, esta base não traz os metadados necessários: use outra fonte documentada ou dados sintéticos identificados como tais, respeitando a exigência de ao menos um experimento real.

### Separe o experimento de erro da avaliação principal

Use somente as linhas de **treino + validação da B2** para criar partições internas da Parte C. O teste principal da D deve continuar reservado. Nomeie as variáveis com o prefixo `c_` para não substituir, sem perceber, os dados, estatísticas ou modelo da D.

Para cada forma:

1. Descreva qual informação deveria estar protegida e em qual etapa ela atravessa a fronteira.
2. Defina o conjunto de desenvolvimento interno, a avaliação interna, o modelo e a métrica antes de comparar.
3. Execute o procedimento incorreto, identificando-o no código.
4. Execute a versão corrigida. Mantenha comparáveis dados, semente e configuração sempre que possível. Se a correção exige mudar a partição, explique a consequência para a comparação.
5. Registre os dois valores e a diferença. Para acurácia entre 0 e 1, a diferença em pontos percentuais é `100 * (errado - correto)`.
6. Explique por que os resultados sustentam, ou limitam, sua interpretação. Uma pequena diferença ou uma diferença de sinal inesperado não invalida automaticamente a existência do vazamento.

| Forma | Dados usados | Métrica | Incorreto | Corrigido | Diferença/unidade | Informação indevida |
|---|---|---|---|---|---|---|
| preencher | preencher | preencher | medir | medir | calcular | explicar |

Se escolher a forma 7, reserve **quatro conjuntos internos desde o início**. Faça a seleção indevida entre os 50 candidatos pelo teste interno e avalie o escolhido no quarto conjunto, intocado até então. Compare com uma seleção orientada pela validação interna, usando o mesmo orçamento de candidatos. Nenhum desses conjuntos substitui o teste principal da D. Não use o quarto conjunto para corrigir a busca depois de ver o resultado.

**Entregue:** dois experimentos executados, tabela e um parágrafo explicativo por forma. Não replique observações ou tente sementes sucessivas apenas para produzir um efeito maior. Se introduzir uma coluna indevida artificialmente para demonstrar o mecanismo, declare exatamente essa transformação.

## 7. Parte D — Métricas e recomendação de limiar · 0,30

Use as partições principais da B2 e a classe positiva `quality >= 7`. O classificador principal deve ser uma **rede binária pequena em PyTorch**, com parada antecipada. O arquivo `apoio.py` oferece uma arquitetura de referência, funções de treino por época, perda e escores. Uma regressão logística adicional é opcional e não substitui a D7.

Comece com uma rede 11 → 16 → 1, ReLU, Adam e `BCEWithLogitsLoss` sem ponderação. Esses valores são uma referência de baixo custo, não uma promessa de desempenho. Registre qualquer mudança e faça escolhas somente na validação. Converter logit em escore com sigmoide não garante calibração.

### D7 primeiro: aprenda e restaure o melhor estado

Implemente parada antecipada monitorando a **perda de validação**. Defina antes o máximo de épocas, a paciência e a melhora mínima; sugestões: 100 épocas, paciência 10 e melhora mínima de 0,0001. A cada época, treine, avalie sem gradientes e decida se houve melhora.

Ao melhorar, guarde uma **cópia independente dos pesos**, a época e a perda. Uma referência mutável ao `state_dict` pode continuar mudando durante o treino. Ao esgotar a paciência, pare e restaure os pesos guardados antes de gerar escores para decidir o limiar.

**Mostre:** curvas de perda de treino e validação, época de parada, melhor época, melhor perda e confirmação da restauração. Se o máximo de épocas chegar primeiro, informe isso; não diga que a paciência disparou quando não disparou. Não monitore a perda do teste durante esse laço.

### D4. Declare a decisão e os custos

No contexto didático de selecionar amostras de alta nota, defina o que ocorre quando o sistema prevê positivo e quando prevê negativo. Então descreva:

- **Falso positivo:** selecionar como alta nota uma observação com nota abaixo de 7.
- **Falso negativo:** deixar de selecionar uma observação com nota pelo menos 7.

Proponha `C_FP > 0` e `C_FN > 0`, numa unidade consistente. Podem ser unidades de esforço, pontos de custo ou valores hipotéticos. Esta base **não fornece preços, perdas ou custos operacionais reais**. Escreva explicitamente que são hipóteses do exercício e justifique a relação entre eles. Para simplificar, considere custo zero para acertos. A qualidade da justificativa vale mais que a magnitude escolhida.

### D5. Escolha o limiar na validação

Com o melhor modelo restaurado, obtenha escores na validação. Para cada limiar candidato, classifique positivo quando `escore >= limiar` e calcule:

```text
custo = C_FP × número de falsos positivos + C_FN × número de falsos negativos
```

Avalie os escores únicos e os extremos que representam todas as previsões positivas ou negativas; se usar uma grade, declare a resolução e que ela é uma aproximação. Declare também como desempatar limiares de custo igual. Compare o escolhido com 0,5 **na mesma validação** e informe custo total e custo médio por observação.

Fixe a regra de desempate, o modelo, as características, as estatísticas e o limiar antes da avaliação final. A mesma validação orienta a parada e o limiar neste laboratório; reconheça esse uso de desenvolvimento. Seus números de validação não são a estimativa independente final.

### D6. Compare a busca com a fórmula

Calcule `C_FP / (C_FP + C_FN)` e compare com o corte encontrado. A fórmula assume probabilidades apropriadas à população, custos de erro constantes e custo zero para acertos. Sua validade não decorre apenas do nome `predict_proba` ou da sigmoide.

Investigue diferenças: amostra pequena, escores mal calibrados, grade de busca, pesos de classe ou reamostragem. Um diagrama de calibração na validação e o Brier score podem apoiar a discussão, sem provar calibração perfeita. Não ajuste o modelo depois de olhar o teste para fazer as duas respostas coincidirem.

### D1 e D2. Abra a avaliação final e produza a tabela

Use o teste principal depois de fixar as escolhas. Gere os escores uma vez e calcule, com a regra escolhida: matriz de confusão, acurácia, precisão, revocação, F1 e AUC-ROC. Para a curva precisão–revocação, informe a **área trapezoidal AUC-PR**. Se também usar `average_precision_score`, rotule-a **AP**: ela não é, em geral, a mesma área.

Apresente a proporção positiva do conjunto avaliado. A matriz deve ter rótulos e orientação explícitos: linhas reais e colunas previstas, na ordem 0 e 1. Calcule AUCs com **escores**, não com as classes após o corte. Declare a convenção usada se nenhuma previsão positiva for produzida.

Construa o classificador trivial usando a **classe majoritária identificada no treino** e avalie-o no mesmo teste. Compare a acurácia e a capacidade de encontrar positivos. Faça a tabela de 0,5 e do limiar previamente fixado; a comparação no teste serve para relatar essas duas regras já definidas, não para escolher uma terceira.

| Regra | Limiar | Acurácia | Precisão | Revocação | F1 | AUC-ROC | AUC-PR | Custo/observação |
|---|---|---|---|---|---|---|---|---|
| Trivial do treino | não se aplica | medir | medir | medir | medir | justificar | justificar | medir |
| Rede: padrão | 0,5 | medir | medir | medir | medir | medir | medir | medir |
| Rede: escolhido na validação | fixado | medir | medir | medir | medir | medir | medir | medir |

As AUCs da rede são iguais nas duas linhas quando os escores são os mesmos: o limiar muda as classes, não o ordenamento. Para um baseline constante, declare qual escore constante usou ou marque as áreas como não reportadas; não crie uma curva inexistente a partir de uma única matriz.

### D3. Desenhe as duas curvas

Plote ROC e precisão–revocação lado a lado, com nomes dos eixos e áreas calculadas. Na ROC, a referência aleatória é a **diagonal** de (0,0) a (1,1), cuja área é 0,5; não uma linha horizontal em 0,5. Na PR, use a linha horizontal correspondente à prevalência positiva **do teste**.

### D8. Escreva para quem decide

Conclua com um parágrafo: qual ação o modelo apoia, qual limiar foi fixado, quais métricas e custos foram observados no teste, qual erro merece atenção e quais limitações continuam abertas. Identifique custos como hipotéticos. Não transforme desempenho neste arquivo em garantia para produtores, regiões ou períodos que não foram avaliados.

**Entregue na D:** histórico de treino, restauração, matriz de custos, busca na validação, comparação com a fórmula, duas matrizes de confusão da rede, tabela final, curvas e recomendação. Mantenha separadas a evidência usada para escolher e a evidência usada para avaliar.

## 8. Como a nota é composta

Os valores máximos de A, B, C e D são 0,20, 0,20, 0,30 e 0,30. Dentro de cada parte, a correção considera estas dimensões; elas não constituem pontos adicionais:

| Dimensão | Peso | Evidência |
|---|---|---|
| Execução | 25% | Código roda do zero e mantém saídas visíveis. |
| Correção técnica | 30% | Partições e métricas corretas, mecanismos de vazamento demonstrados. |
| Análise | 30% | Explicação das causas, limitações e escolhas. |
| Comunicação | 15% | Texto legível, tabelas e gráficos identificados. |

As regras de penalização do laboratório continuam aplicáveis:

| Situação | Efeito previsto |
|---|---|
| Notebook não executa | Execução = 0 e −50% nas demais dimensões. |
| Vazamento na própria Parte D não detectado | −40%. |
| Vazamento detectado e documentado pelo estudante | Sem penalidade por essa descoberta; registrar a correção. |
| Métrica de treino reportada como resultado de avaliação | −30%. |
| Experimento inflado para parecer dramático | −20% na dimensão Análise. |
| Ausência de `USO-DE-IA.md` | Nota zero. |
| Atraso de até 48 horas | −20%. |

**Bônus de até 0,1:** demonstrar uma oitava forma de vazamento, com medição e explicação, ou documentar um mecanismo de vazamento em uma base pública conhecida. Repetir o alerta de duplicatas já fornecido neste pacote, sem investigação adicional, não constitui uma descoberta nova.

## 9. Checklist antes de enviar

- B2: todas as linhas cobertas, nenhuma linha ou perfil compartilhado, ambas as classes nos conjuntos.
- A: estatísticas aprendidas no treino; tipos e formas corretos; quatro tempos medidos; erro de BatchNorm documentado.
- C: duas formas, ao menos uma real e uma entre 2/4/7; comparação e diferença com unidade; teste principal preservado.
- D: melhor estado restaurado; limiar escolhido na validação; teste usado depois das escolhas; matriz, curvas e custos identificados.
- Relatório: nenhum valor inventado; resultados pequenos ou desfavoráveis também explicados.
- Entrega: notebook reiniciado e executado, README preenchido, declaração de IA, fonte dos dados e versões.
- Arquivos de resposta: nenhum marcador TODO tratado como conclusão ou célula “pendente” como evidência.

## 10. Dúvidas frequentes e apoio

**Preciso baixar a base do Kaggle?** Não. O mesmo conjunto tem uma fonte original pública; o CSV já está incluído, com atribuição. A indicação do Kaggle serve para conhecer sua página e outros trabalhos, não para copiar resultados.

**Posso usar o banco do meu projeto?** A atividade está preparada para a base comum, sem exigir um domínio particular. Uma substituição deve atender a todos os requisitos e ser combinada com o professor; não é necessária para concluir este material.

**Por que a acurácia trivial é alta?** Calcule as proporções. A predominância de uma classe pode tornar esse indicador pouco informativo sozinho. Explique com seus resultados, não com a frase pronta “acurácia não presta”.

**O vazamento mudou pouco a métrica. Fiz errado?** Não necessariamente. Investigue o mecanismo e reporte a medição. Não é exigido obter a diferença mostrada nos slides.

**Pipeline resolve tudo?** Ele organiza transformações e pode evitar ajustes no conjunto errado quando usado corretamente. Não corrige sozinho grupos misturados, uma coluna com informação do futuro ou escolhas feitas olhando o teste.

**Posso usar IA?** Registre ferramenta, finalidade, trecho aproveitado e como verificou. Se não usou, declare isso no arquivo obrigatório. Você deve conseguir explicar cada linha entregue.

**A base não apareceu no Colab.** Confira o diretório atual, a extração da pasta `classroom-lab-02/` e a existência de `dados/winequality-red.csv`. A leitura usa separador `;`, não vírgula.

**O DataLoader travou.** Teste `num_workers=0`, confira se o kernel usa o ambiente correto e registre a condição. Evite executar criação de processos em paralelo no Windows sem proteger o ponto de entrada de scripts.

Fontes de apoio: [Dataset/DataLoader no PyTorch](https://docs.pytorch.org/docs/stable/data.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [métricas](https://scikit-learn.org/stable/modules/model_evaluation.html), [escolha de limiar](https://scikit-learn.org/stable/modules/classification_threshold.html) e [cuidados com vazamento](https://scikit-learn.org/stable/common_pitfalls.html).
