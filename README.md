# Steepest Descent Method

Este proyecto implementa el método de descenso más pronunciado (Steepest Descent) para resolver problemas de optimización. El método se utiliza para encontrar el mínimo de una función objetivo mediante iteraciones que ajustan los valores iniciales en la dirección del gradiente negativo.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal utilizado para implementar el algoritmo.
- **NumPy**: Biblioteca para operaciones matemáticas y manejo de arrays.
- **SciPy**: Utilizada para calcular aproximaciones al gradiente de las funciones objetivo.
- **Pandas**: Para la manipulación y exportación de datos en formato tabular.
- **Tabulate**: Para mostrar los resultados en formato de tabla en la consola.
- **Excel**: Los resultados de las iteraciones se exportan a archivos Excel para un análisis más detallado.

## Funcionalidad

El proyecto evalúa el método de descenso más pronunciado en cuatro funciones objetivo diferentes. Los resultados de cada iteración se almacenan y exportan para su análisis. A continuación, se describen las principales características:

1. **Funciones Objetivo**: Se incluyen cuatro funciones objetivo diferentes, cada una con características únicas para evaluar el rendimiento del método.
2. **Tamaños de Paso (Step Sizes)**: Se evalúan diferentes tamaños de paso (constantes y variables) para observar su impacto en la convergencia del método.
3. **Exportación de Resultados**: Los datos de cada iteración, incluyendo el punto actual, el gradiente y la norma del gradiente, se exportan a archivos Excel organizados por función y tamaño de paso.
4. **Análisis de Datos**: Los resultados se presentan en formato tabular en la consola y se guardan en archivos Excel para facilitar el análisis posterior.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **`steepest_descent.py`**: Archivo principal que contiene la implementación del método de descenso más pronunciado.
- **Directorios por Función**: Cada función objetivo tiene su propio directorio donde se almacenan los resultados de las iteraciones con diferentes tamaños de paso.
  - `function_1/`
  - `function_2/`
  - `function_3/`
  - `function_4/`
- **`variable_step_size/`**: Directorio que contiene los resultados de las iteraciones con tamaños de paso variables.

## Trabajo con Datos

El proyecto utiliza datos de manera intensiva para estudiar y comprender el comportamiento del método de descenso más pronunciado:

- **Generación de Datos**: En cada iteración, se calculan y almacenan los siguientes datos:
  - Número de iteración.
  - Punto actual \(x_k\).
  - Gradiente \(\nabla f\).
  - Norma del gradiente \(||\nabla f||\).
- **Exportación a Excel**: Los datos se exportan a archivos Excel para cada combinación de función y tamaño de paso, lo que permite un análisis detallado y visualización en herramientas como Microsoft Excel.
- **Visualización en Consola**: Los datos también se presentan en formato tabular en la consola para una revisión rápida.

## Cómo Ejecutar el Proyecto

1. Asegúrate de tener instaladas las dependencias necesarias:
   ```bash
   pip install numpy scipy pandas tabulate
   ```
2. Ejecuta el archivo principal:
   ```bash
   python steepest_descent.py
   ```
3. Los resultados se generarán en los directorios correspondientes y se exportarán a archivos Excel.

---
