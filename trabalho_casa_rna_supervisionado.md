# Trabalho para Casa: Aprendizado Supervisionado com Redes Neurais Artificiais
## Previsao de Qualidade de Vinhos a partir de Propriedades Fisico-Quimicas

**Disciplina:** Inteligencia Artificial  
**Professor:** Marcio Andrey Roselli  
**Instituicao:** Universidade do Distrito Federal (UnDF)  
**Turma:** Engenharia de Software  
**Modalidade:** Individual  
**Prazo de Entrega:** 28 dias apos a data de disponibilizacao  


## 1. Descricao do Problema

O dataset **Wine Quality** (Cortez et al., 2009) contem analises fisico-quimicas de amostras de vinho tinto portugues da regiao do Minho (variante Vinho Verde). Cada amostra possui 11 atributos de entrada e uma nota de qualidade atribuida por avaliadores sensoriais (escala de 0 a 10).

O dataset e publico e pode ser obtido de duas formas:

```python
# Opcao 1: direto do UCI Machine Learning Repository
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Opcao 2: via scikit-learn (requer versao >= 1.3)
# from sklearn.datasets import fetch_openml
# df = fetch_openml("wine-quality-red", as_frame=True).frame
```

### Atributos de Entrada (11)

| Atributo | Descricao | Unidade |
|----------|-----------|---------|
| fixed acidity | Acidez fixa (acido tartarico) | g/dm3 |
| volatile acidity | Acidez volatil (acido acetico) | g/dm3 |
| citric acid | Acido citrico | g/dm3 |
| residual sugar | Acucar residual | g/dm3 |
| chlorides | Cloretos (cloreto de sodio) | g/dm3 |
| free sulfur dioxide | Dioxido de enxofre livre | mg/dm3 |
| total sulfur dioxide | Dioxido de enxofre total | mg/dm3 |
| density | Densidade | g/cm3 |
| pH | pH | - |
| sulphates | Sulfatos (sulfato de potassio) | g/dm3 |
| alcohol | Teor alcoolico | % vol. |

### Variavel Alvo

`quality`: nota inteira de 0 a 10 (na pratica, as amostras variam entre 3 e 8).

**Tarefa:** O aluno devera transformar o problema em classificacao binaria:

```python
# Vinho "bom" (quality >= 7) vs. "regular" (quality < 7)
df["alvo"] = (df["quality"] >= 7).astype(int)
```

Isso gera um problema desbalanceado (aproximadamente 13% da classe positiva), o que faz parte do desafio.


## 2. Objetivos

1. Construir, treinar e avaliar uma rede neural feedforward em PyTorch, TensorFlow/Keras ou JAX para classificacao binaria.
2. Tratar um dataset desbalanceado com pelo menos duas estrategias distintas.
3. Comparar a rede neural com um modelo de referencia simples (baseline).
4. Produzir um relatorio tecnico com analise critica dos resultados.


## 3. Requisitos Obrigatorios

O trabalho deve conter, no minimo, as seguintes etapas:

### 3.1 Analise Exploratoria

- Estatisticas descritivas (media, desvio padrao, min, max) dos 11 atributos.
- Distribuicao da variavel alvo original (histograma das notas) e da variavel binaria.
- Identificacao de atributos com maior correlacao com a variavel alvo.
- Verificacao de valores ausentes e outliers.

### 3.2 Pre-processamento

- Divisao treino/teste (80/20) com estratificacao pela variavel alvo.
- Normalizacao dos atributos (StandardScaler ou MinMaxScaler), aplicada apenas no conjunto de treino e depois transformada no teste.
- Uma estrategia para tratar o desbalanceamento.

### 3.3 Rede Neural

- Minimo de 2 camadas ocultas.
- Justificar a escolha de: funcao de ativacao, numero de nos por camada, otimizador, taxa de aprendizagem e numero de epocas.
- Plotar curvas de perda (treino e validacao) ao longo das epocas.
- Implementar pelo menos uma tecnica de regularizacao (Dropout, weight decay ou early stopping).

### 3.4 Analise Critica

Responder as seguintes questoes no relatorio:

**Q1.** A rede neural superou o baseline de regressao logistica? Se a diferenca foi pequena, por que isso pode ocorrer considerando a natureza do dataset (11 atributos, 1599 amostras)?

**Q2.** Analise o trade-off entre precision e recall para a classe "bom". Em que cenario pratico (ex.: selecao de vinhos para exportacao vs. controle de qualidade interno) cada metrica seria prioritaria?

**Q3.** Considerando o Teorema da Aproximacao Universal, uma rede com uma unica camada oculta suficientemente grande deveria ser capaz de resolver este problema. Na pratica, por que redes com duas ou tres camadas menores tendem a funcionar melhor do que uma unica camada muito larga?

**Q4.** Se este modelo fosse implantado em uma cooperativa vinicola para classificacao automatica, quais validacoes adicionais seriam necessarias antes do uso em producao? Considere aspectos de robustez, distribuicao dos dados e responsabilidade sobre erros.

## 4. Estrutura do Relatorio

O relatorio deve ser entregue em PDF com a seguinte organizacao:

1. **Introducao** (0,5 pagina): descricao do problema e do dataset.
2. **Metodologia** (1-2 paginas): pre-processamento, arquitetura da rede, hiperparametros, estrategias de desbalanceamento.
3. **Resultados** (2-3 paginas): tabelas comparativas, graficos (curvas de aprendizagem, matriz de confusao, ROC, Precision-Recall).
4. **Discussao** (1-2 paginas): respostas as questoes Q1-Q5.
5. **Conclusao** (0,5 pagina).
6. **Referencias**.

Extensao total: 6 a 8 paginas (sem contar apendices).

Entregar tambem o notebook (.ipynb) executavel como apendice.