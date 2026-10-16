# Detecção de Fraude em Cartão de Crédito

Projeto de machine learning para detectar fraudes em transações reais de cartão
de crédito, com foco no tratamento do desbalanceamento extremo entre as classes.

## 1. O problema

A base contém **284.807 transações**, das quais apenas **492 são fraudes** —
cerca de **0,173%** do total. Em outras palavras, **99,827%** das transações são
normais.

Por isso a **acurácia engana**: um modelo que responde "não é fraude" para tudo
acertaria 99,8% dos casos, mas teria **recall de 0** para a fraude — ou seja,
não pegaria nenhuma fraude de verdade. As métricas que importam aqui são:

- **Recall (classe 1):** de todas as fraudes reais, quantas o modelo pegou.
- **Precisão (classe 1):** das que o modelo marcou como fraude, quantas eram fraude.
- **F1 (classe 1):** equilíbrio entre as duas.

A métrica principal escolhida foi o **recall da fraude**, já que deixar passar
uma fraude costuma ser mais caro que bloquear uma transação legítima.

## 2. Preparação dos dados

- Carregamento do dataset via link (sem versionar o CSV no repositório).
- Criação da variável `log_amount` (log1p do valor), pois `Amount` tem cauda longa.
- Padronização de `Time` e `Amount` com `StandardScaler` (as V1–V28 já vêm
  padronizadas pelo PCA original).
- Separação treino/teste **estratificada** (70/30), garantindo a mesma
  proporção de fraudes nos dois conjuntos:
  - Treino: 99,827% normais / 0,173% fraudes
  - Teste: 99,827% normais / 0,173% fraudes
- Peso das classes (`class_weight='balanced'` na Regressão Logística,
  `balanced_subsample` no Random Forest e `scale_pos_weight` no XGBoost),
  como alternativa mais simples ao undersampling/oversampling.

## 3. Comparação dos modelos

| Modelo | Recall (fraude) | Precisão (fraude) | F1 (fraude) |
|---|---|---|---|
| Regressão Logística (baseline) | **0,8784** | 0,0653 | 0,1216 |
| Random Forest | 0,7095 | **0,9722** | 0,8203 |
| XGBoost | 0,7838 | 0,9062 | 0,8406 |

**Leitura dos resultados:**
- A Regressão Logística tem o **maior recall** (pega 87,8% das fraudes), mas
  com **precisão muito baixa** (só 6,5% do que ela marca como fraude é fraude
  de verdade) — muitos falsos positivos.
- Random Forest e XGBoost têm precisão altíssima, mas **deixam passar mais
  fraudes** (recall menor).
- Nenhum modelo "ganha" em tudo: existe um trade-off claro entre recall e
  precisão, o que justifica ajustar o limiar de decisão (seção 4).

> Cole aqui as figuras das curvas ROC e Precision-Recall geradas no notebook.

## 4. Limiar de decisão

O `predict()` usa 0,5 por padrão, o que é ruim em classes desbalanceadas.
Ajustei o limiar para **0,29**, escolhido por **maximizar o F1**. Resultado com
o limiar ajustado (Random Forest):

| Métrica (classe 1) | Limiar 0,5 (padrão) | Limiar 0,29 (ajustado) |
|---|---|---|
| Recall | 0,7095 | **0,7770** |
| Precisão | 0,9722 | 0,9350 |
| F1 | 0,8203 | **0,8487** |

O ajuste **aumentou o recall em ~7 pontos percentuais** (de 70,9% para 77,7%),
com pequena perda de precisão — um trade-off que faz sentido no contexto de
fraude, onde deixar passar uma fraude custa mais caro que um falso positivo.

## 5. O que o SHAP mostrou

As variáveis que mais pesaram nas decisões do modelo foram, em ordem:

**V14, V4, V12, V10, V17** (e, em seguida, V11, V3, V7, V16, V2).

Como essas colunas vêm de PCA, não têm nome interpretável. Mas o padrão do
gráfico mostra que:

- Valores **baixos** de V14, V12 e V10 empurram fortemente para **fraude**.
- Valores **altos** de V4 e V11 também empurram para **fraude**.
- `log_amount` e `Amount_scaled` aparecem com peso menor, mas ainda relevante.

> Cole aqui o `shap.summary_plot` gerado no notebook.

## 6. O que mudei em relação ao que a Expert fez

- **Pesos das classes em vez de reamostragem:** usei `class_weight='balanced'`
  no LogReg, `balanced_subsample` no Random Forest e `scale_pos_weight` no
  XGBoost, em vez de SMOTE/undersampling — mais simples e sem risco de vazamento.
- **Ajuste de limiar no Random Forest:** testei o limiar ótimo por F1 (0,29) e
  comparei com o padrão 0,5, mostrando o ganho em recall.
- **SHAP sobre amostra:** o cálculo completo do SHAP no Random Forest com 200
  árvores inviabilizava o tempo de execução no Colab gratuito (>25 min). Usei
  uma **amostra de 1.000 transações** do teste, o que mantém a tendência geral
  e roda em poucos minutos.
- **Três modelos comparados** (LogReg, RF, XGBoost) em vez de apenas um, para
  mostrar o trade-off recall × precisão.

## 7. Como rodar

1. Abra o notebook no Google Colab.
2. Rode todas as células em ordem (**Ambiente de execução > Executar tudo**).
3. O dataset é carregado automaticamente por link:
   `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

## Tecnologias

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, SHAP, Matplotlib.
