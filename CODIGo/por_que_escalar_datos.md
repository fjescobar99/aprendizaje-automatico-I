# ¿Por qué es necesario escalar los datos?

Hay tres razones principales:

---

## 1. Las variables están en escalas muy diferentes

En el dataset `Default`:
- `balance` va de ~0 a ~2600 (dólares de saldo)
- `income` va de ~700 a ~73,000 (dólares de ingreso)
- `student_Yes` es 0 o 1

Sin escalar, el modelo podría interpretar erróneamente que `income` es "más importante" simplemente porque sus números son más grandes, no porque realmente explique mejor el default.

---

## 2. La regularización funciona mal sin escalar

La regresión logística en scikit-learn aplica regularización L2 por defecto. La regularización penaliza los coeficientes grandes para evitar overfitting, pero si `income` tiene valores de 50,000 su coeficiente será naturalmente pequeño (ej: 0.00003) y si `balance` tiene valores de 1,000 su coeficiente será mayor. La penalización castigaría de forma injusta a unas variables sobre otras.

> La comparación de coeficientes solo es válida cuando los datos están escalados.

---

## 3. La convergencia del optimizador es más lenta (o falla)

Los algoritmos de optimización (como el gradiente descendente que usan los solvers de scikit-learn) convergen más rápido cuando las variables están en rangos similares. Con escalas muy dispares, el gradiente puede oscilar o tardar muchas más iteraciones.

---

## Resumen

| Sin escalar | Con escalar |
|---|---|
| Coeficientes no comparables | Coeficientes comparables en magnitud |
| Regularización sesgada | Regularización justa entre variables |
| Convergencia lenta | Convergencia más eficiente |

> **Nota:** La variable `student_Yes` no se escala porque ya es binaria (0/1), está naturalmente en la misma escala que los datos transformados.
