# Classificação de Qualidade de Vinhos com Redes Neurais Artificiais

Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial da Universidade do Distrito Federal (UnDF), com o objetivo de classificar vinhos tintos em duas categorias a partir de atributos físico-químicos: `regular` e `bom`.

O trabalho utiliza o dataset `Wine Quality` e compara o desempenho de uma rede neural artificial feedforward com um baseline de regressão logística, adotando uma abordagem experimental completa com análise exploratória, pré-processamento, tratamento de desbalanceamento, treinamento, avaliação e discussão crítica dos resultados.

## Objetivo

Construir um modelo de classificação binária capaz de prever se um vinho deve ser considerado:

- `0` = regular, quando `quality < 7`
- `1` = bom, quando `quality >= 7`

A proposta central do notebook é verificar se uma rede neural consegue superar um modelo linear simples em um conjunto de dados tabular, relativamente pequeno e desbalanceado.

## Visão Geral da Solução

O notebook [`trabalho_rna_classificacao.ipynb`](./trabalho_rna_classificacao.ipynb) implementa o seguinte fluxo:

1. Importação das bibliotecas necessárias.
2. Carregamento do arquivo `winequality-red.csv`.
3. Criação da variável alvo binária `alvo`.
4. Análise exploratória dos dados.
5. Separação entre treino, validação e teste.
6. Padronização das variáveis com `StandardScaler`.
7. Tratamento do desbalanceamento de classes.
8. Treinamento de um baseline com regressão logística.
9. Treinamento de duas versões da rede neural.
10. Comparação dos modelos com múltiplas métricas.
11. Visualização das curvas e gráficos finais.
12. Análise crítica dos resultados.

## Estrutura do Projeto

```text
.
|-- README.md
|-- trabalho_rna_classificacao.ipynb
|-- trabalho_casa_rna_supervisionado.md
`-- winequality-red.csv
```

## Dataset

O projeto utiliza o arquivo [`winequality-red.csv`](./winequality-red.csv), contendo amostras de vinhos tintos descritas por atributos numéricos, como:

- acidez fixa
- acidez volátil
- ácido cítrico
- açúcar residual
- cloretos
- dióxido de enxofre livre e total
- densidade
- pH
- sulfatos
- teor alcoólico

A coluna original `quality` representa a nota do vinho. No notebook, essa variável é transformada em um problema de classificação binária por meio da criação da coluna `alvo`.

## Metodologia

### 1. Análise Exploratória

O notebook realiza uma inspeção inicial dos dados para entender:

- distribuição da variável `quality`
- proporção entre as classes da variável `alvo`
- presença de valores ausentes
- correlação entre os atributos e a classe alvo
- existência de outliers

Os outliers são mantidos no conjunto por poderem representar sinal real do processo químico, e parte de seu impacto é reduzida na etapa de normalização.

### 2. Pré-processamento

Os dados são divididos de forma estratificada em:

- treino: 64%
- validação: 16%
- teste: 20%

Em seguida, é aplicado `StandardScaler`, ajustado apenas no conjunto de treino e reutilizado em validação e teste, evitando vazamento de informação.

### 3. Tratamento do Desbalanceamento

Como a classe positiva (`bom`) é minoritária, o notebook testa duas estratégias:

- `class_weight`: aumenta o peso da classe minoritária durante o treinamento
- `oversampling`: replica amostras da classe minoritária no conjunto de treino

Essa comparação permite avaliar qual técnica oferece melhor equilíbrio entre precisão e recall.

### 4. Baseline

Antes do treinamento da rede neural, é ajustado um modelo de regressão logística com balanceamento de classes. Esse baseline serve como referência para medir se a arquitetura neural realmente agrega valor ao problema.

### 5. Rede Neural Artificial

A arquitetura escolhida é uma rede feedforward para classificação binária, composta por:

- camada de entrada com 11 atributos
- primeira camada oculta com 64 neurônios e ativação `ReLU`
- `Dropout` para regularização
- segunda camada oculta com 32 neurônios e ativação `ReLU`
- novo `Dropout`
- camada de saída com 1 neurônio e ativação `sigmoid`

Além disso, o modelo utiliza:

- otimizador `Adam`
- função de perda `binary_crossentropy`
- regularização `L2`
- `EarlyStopping`
- `ReduceLROnPlateau`

Foram treinadas duas variantes da rede:

- rede neural com `class_weight`
- rede neural com `oversampling`

## Métricas de Avaliação

Os modelos são avaliados com métricas apropriadas para classificação binária e cenários com desbalanceamento:

- acurácia
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC

O critério adotado para selecionar a melhor rede neural é o `PR-AUC` no conjunto de validação, por ser especialmente útil quando a classe positiva é menos frequente.

## Visualizações Geradas

O notebook produz gráficos que ajudam na interpretação do experimento:

- histograma da qualidade original
- distribuição da classe binária
- correlações com a variável alvo
- boxplots por atributo
- curvas de aprendizado
- matriz de confusão
- curva ROC
- curva Precision-Recall
- comparação de F1-score entre os modelos

## Principais Pontos da Análise

O trabalho foi estruturado para responder, de forma crítica, questões como:

- a rede neural superou o baseline de regressão logística?
- qual foi o trade-off entre `precision` e `recall` para a classe positiva?
- por que uma rede com duas camadas menores pode funcionar melhor do que uma camada única muito larga?
- quais validações adicionais seriam necessárias antes de um uso em produção?

Esse enfoque fortalece o caráter analítico do projeto, indo além da simples execução do modelo.

## Como Executar

### Requisitos

Recomenda-se utilizar Python 3.10+ com as bibliotecas:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`

### Execução

1. Mantenha o arquivo `winequality-red.csv` na raiz do projeto.
2. Abra o notebook [`trabalho_rna_classificacao.ipynb`](./trabalho_rna_classificacao.ipynb) em Jupyter Notebook ou JupyterLab.
3. Execute as células em ordem.

O notebook também procura o dataset no caminho `data/winequality-red.csv`, caso essa estrutura seja adotada.

## Resultados Esperados

Ao final da execução, o notebook:

- compara a regressão logística com duas estratégias de treinamento da rede neural
- identifica a melhor rede com base em `PR-AUC`
- apresenta métricas quantitativas e gráficos interpretativos
- fornece material pronto para discussão em relatório ou apresentação

Como os resultados dependem da execução no ambiente local, os valores finais devem ser confirmados diretamente a partir da saída do notebook.

## Contexto Acadêmico

- Disciplina: Inteligência Artificial
- Professor: Marcio Andrey Roselli
- Instituição: Universidade do Distrito Federal (UnDF)
- Curso: Engenharia de Software

## Autor

- Bruno Araújo Sales
