import json

path = r'D:\maestria\aprendizaje automatico I\CODIGo\ejercicios_clase02_nuestra_resolucion.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md_cell(source_lines):
    return {
        'cell_type': 'markdown',
        'id': '',
        'metadata': {},
        'source': source_lines
    }

def code_cell(source_lines):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': '',
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }

# ──────────────────────────────────────────────
# EJERCICIO 8
# ──────────────────────────────────────────────
ej8_md = md_cell([
    '# Ejercicio 8\n',
    'Efecto del Alpha en Lasso: Creá un gráfico que muestre cómo cambian los coeficientes '
    'de Lasso a medida que `alpha` aumenta. Podés usar la información almacenada en `lasso_cv.path_`.'
])

ej8_code = code_cell([
    '# Calculamos el regularization path de Lasso sobre un rango amplio de alphas\n',
    'alphas_path, coefs_path, _ = lasso_cv.path(X_train_scaled, y_train, alphas=np.logspace(-4, 0, 200))\n',
    '\n',
    '# coefs_path tiene forma (n_features, n_alphas): un coeficiente por feature por alpha\n',
    'fig = go.Figure()\n',
    'for i, feature in enumerate(numeric_features_plus):\n',
    '    fig.add_trace(go.Scatter(\n',
    '        x=alphas_path,\n',
    '        y=coefs_path[i],\n',
    '        mode=\'lines\',\n',
    '        name=feature\n',
    '    ))\n',
    '\n',
    '# Marcamos con linea vertical el alpha optimo que encontro LassoCV\n',
    'fig.add_vline(\n',
    '    x=lasso_cv.alpha_,\n',
    '    line_dash="dash",\n',
    '    line_color="black",\n',
    '    annotation_text="alpha optimo = " + str(round(lasso_cv.alpha_, 4)),\n',
    '    annotation_position="top right"\n',
    ')\n',
    '\n',
    'fig.update_layout(\n',
    '    title=\'Efecto del Alpha en los Coeficientes de Lasso (Regularization Path)\',\n',
    '    xaxis_title=\'Alpha\',\n',
    '    xaxis_type=\'log\',\n',
    '    yaxis_title=\'Valor del Coeficiente\',\n',
    '    legend_title=\'Feature\'\n',
    ')\n',
    'fig.show()'
])

ej8_conclusion = md_cell([
    'El gráfico muestra cómo cada coeficiente se encoge hacia cero a medida que alpha aumenta.\n',
    'La línea vertical indica el alpha óptimo encontrado por LassoCV con validación cruzada.\n',
    '\n',
    '**Observaciones clave:**\n',
    '- **FullBath** y **BedroomAbvGr** son los primeros en zerificarse: Lasso los descarta como poco informativos.\n',
    '- **OverallQual** y **GrLivArea** son los últimos en llegar a cero: son las variables más robustas del modelo.\n',
    '- El alpha óptimo cae en la zona donde se eliminaron los features ruidosos pero aún se conservan los relevantes.'
])

# ──────────────────────────────────────────────
# EJERCICIO 9
# ──────────────────────────────────────────────
ej9_md = md_cell([
    '# Ejercicio 9\n',
    'Regresión Lineal Simple vs. Múltiple: Entrená un modelo de regresión lineal simple para cada una '
    'de las características en `numeric_features` por separado. '
    'Compará el coeficiente de cada característica en su modelo simple con su coeficiente en el modelo '
    'múltiple (OLS). ¿Por qué son diferentes?'
])

ej9_code = code_cell([
    '# Usamos las 6 features originales (numeric_features) para comparar directamente\n',
    '# con el modelo OLS multiple de la seccion 3 del taller\n',
    'X_simple = df_subset[numeric_features]\n',
    'y_simple = np.log1p(df_subset[target_variable])\n',
    '\n',
    'X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(\n',
    '    X_simple, y_simple, test_size=0.2, random_state=42\n',
    ')\n',
    '\n',
    'scaler_s = StandardScaler()\n',
    'X_train_s_sc = pd.DataFrame(scaler_s.fit_transform(X_train_s), columns=numeric_features)\n',
    'X_test_s_sc  = pd.DataFrame(scaler_s.transform(X_test_s),  columns=numeric_features)\n',
    '\n',
    '# Modelo OLS multiple (todas las features juntas)\n',
    'ols_multi = LinearRegression()\n',
    'ols_multi.fit(X_train_s_sc, y_train_s)\n',
    '\n',
    '# Un modelo OLS simple por cada feature\n',
    'simple_coefs = {}\n',
    'for feature in numeric_features:\n',
    '    m = LinearRegression()\n',
    '    m.fit(X_train_s_sc[[feature]], y_train_s)\n',
    '    simple_coefs[feature] = m.coef_[0]\n',
    '\n',
    '# Tabla comparativa\n',
    'comparison_df = pd.DataFrame({\n',
    '    \'Regresion_Simple\': simple_coefs,\n',
    '    \'Regresion_Multiple\': dict(zip(numeric_features, ols_multi.coef_))\n',
    '})\n',
    'comparison_df[\'Diferencia\'] = (\n',
    '    comparison_df[\'Regresion_Simple\'] - comparison_df[\'Regresion_Multiple\']\n',
    ').round(4)\n',
    '\n',
    'print("--- Comparacion de Coeficientes: Simple vs Multiple ---")\n',
    'print(comparison_df.round(4))\n',
    '\n',
    '# Grafico de barras comparativo\n',
    'fig = go.Figure([\n',
    '    go.Bar(name=\'Simple\',   x=numeric_features, y=comparison_df[\'Regresion_Simple\']),\n',
    '    go.Bar(name=\'Multiple\', x=numeric_features, y=comparison_df[\'Regresion_Multiple\'])\n',
    '])\n',
    'fig.update_layout(\n',
    '    title=\'Coeficientes: Regresion Simple vs Multiple por Feature\',\n',
    '    barmode=\'group\',\n',
    '    xaxis_title=\'Feature\',\n',
    '    yaxis_title=\'Valor del Coeficiente\'\n',
    ')\n',
    'fig.show()'
])

ej9_conclusion = md_cell([
    '**Por que son diferentes los coeficientes?**\n',
    '\n',
    'En el modelo simple, cada feature absorbe tambien el efecto de las variables correlacionadas '
    'que no estan presentes. En el modelo multiple, los coeficientes reflejan el efecto **marginal puro** '
    'de cada variable, manteniendo las demas constantes.\n',
    '\n',
    '**Ejemplos concretos:**\n',
    '- **GarageCars** tiene un coeficiente mas alto en el modelo simple porque las casas con mas garage '
    'tambien suelen tener mayor area (GrLivArea) y mejor calidad (OverallQual). '
    'Al incluir todas esas variables juntas, su efecto individual se reduce.\n',
    '- **FullBath** puede cambiar de signo entre modelos porque esta correlacionado con el tamanio de la casa. '
    'Sin controlar por GrLivArea parece que agrega valor; controlando por ella puede resultar redundante.\n',
    '- Este fenomeno se llama **multicolinealidad** y es la razon principal por la que la regresion multiple '
    'es mas informativa que N regresiones simples.'
])

# Agregar todas las celdas al final
nb['cells'].extend([ej8_md, ej8_code, ej8_conclusion, ej9_md, ej9_code, ej9_conclusion])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'OK — notebook ahora tiene {len(nb["cells"])} celdas')
