# Dicionário da base comum

O CSV usa **ponto e vírgula** como separador e ponto como separador decimal. Leia com pd.read_csv(caminho, sep=';'). Os nomes têm espaços; preserve-os.

| Coluna original | Leitura em português | Papel e interpretação |
|---|---|---|
| fixed acidity | Acidez fixa | Medida numérica de ácidos pouco voláteis. |
| volatile acidity | Acidez volátil | Medida numérica de ácidos voláteis. |
| citric acid | Ácido cítrico | Medida numérica do componente. |
| residual sugar | Açúcar residual | Medida numérica de açúcar remanescente. |
| chlorides | Cloretos | Medida numérica do componente. |
| free sulfur dioxide | Dióxido de enxofre livre | Medida numérica da fração livre. |
| total sulfur dioxide | Dióxido de enxofre total | Medida numérica total. |
| density | Densidade | Medida física numérica. |
| pH | pH | Índice numérico de acidez. |
| sulphates | Sulfatos | Medida numérica do componente. |
| alcohol | Álcool | Medida numérica do teor de álcool. |
| quality | Nota sensorial | Origem do alvo; não usar como entrada. |

Não é necessário conhecer enologia para fazer o laboratório. As 11 primeiras colunas são entradas; a última é uma nota ordinal transformada em alvo binário. Mantemos as escalas originais; não converta unidades com base em suposições.

## Conferência do arquivo incluído

- 1599 linhas e 12 colunas originais: 11 características e uma nota.
- 0 células ausentes.
- Classe positiva (quality >= 7): 217 linhas (13,57%).
- Classe negativa: 1382 linhas.
- 240 linhas repetidas além da primeira ocorrência de cada linha completa.
- 1359 perfis distintos nas 11 características; 240 ocorrências adicionais.
- 0 perfis idênticos com alvos binários conflitantes neste arquivo.

As contagens descrevem a base inteira, não o resultado de um classificador. Informe também a distribuição da partição avaliada.

## Campos derivados pelo apoio

| Campo | Para que serve | Entra no modelo? |
|---|---|---|
| alvo | Resposta 1 para nota pelo menos 7, senão 0. | Não; é y. |
| id_linha | Conferir cobertura e interseção das linhas. | Não. |
| grupo_perfil | Manter medidas idênticas na mesma partição. | Não. |

grupo_perfil é um código das características, não identidade de produtor, pessoa, safra ou vinho físico. Perfis idênticos não provam que duas observações são a mesma entidade. O arquivo não informa data ou produtor. Registre essas limitações.

Veja a atribuição em FONTE-E-LICENCA.md.
