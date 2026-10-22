# Pattern Lab

Explorador local de patrones textiles construido sobre la biblioteca procedural de [AdversarialPatternsLab de Roni Bandini](https://github.com/ronibandini/AdversarialPatternsLab).

Este fork conserva la licencia GPL-3.0 y los avisos de autoría existentes. Esta primera entrega se enfoca en explorar diseños, visualizar repeticiones y probar una estampa sobre maquetas 2D. La evaluación frente a modelos de detección queda para una etapa posterior.

## Pattern Lab textil — primera versión

- Explorar los 36 patrones procedurales que ya existen en la biblioteca.
- Filtrar por familia y seleccionar una semilla reproducible.
- Generar rondas de variantes sin reemplazar el generador original.
- Cambiar el tamaño visual del motivo y ver el mosaico repetido.
- Previsualizar la textura en dos maquetas vectoriales locales: camiseta y buzo con capucha.
- Descargar el motivo como PNG y sus parámetros reproducibles como JSON.
- Ejecutar localmente en Windows o Linux, sin servicios de IA, API externas ni conexión obligatoria a Internet.

La maqueta es una visualización 2D para estudiar la escala y la distribución del estampado. No simula pliegues, caída de la tela, costuras físicas ni iluminación real.

## Requisitos e instalación

Se necesita Python 3.9 o más reciente, NumPy y OpenCV.

### Windows (PowerShell)

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-pattern-lab.txt
    python app.py

### Linux / macOS

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-pattern-lab.txt
    python app.py

El navegador puede abrirse automáticamente. Si no ocurre, abrí http://127.0.0.1:8765.

El servidor escucha únicamente en 127.0.0.1 por defecto y no publica la aplicación en la red. Para detenerlo, presioná Ctrl+C en la terminal donde está ejecutándose.

Si el puerto 8765 ya está ocupado, se puede usar otro:

    python app.py --port 8766

Para ejecutar sin intentar abrir el navegador:

    python app.py --no-browser

## Semillas y exportación

La combinación de nombre de patrón, semilla y tamaño base identifica una variante reproducible en el mismo entorno de dependencias. La escala de visualización cambia el tamaño del motivo en la repetición y en las maquetas; no modifica los parámetros internos del generador.

- **PNG:** descarga el motivo generado en su tamaño base, sin mosaico ni prenda.
- **JSON:** incluye el nombre del patrón, familia, semilla, tamaño base, escala de visualización, descripción y hash SHA-256 de los píxeles generados.

La exportación actual no guarda capturas de las maquetas ni un patrón listo para impresión industrial. Antes de fabricar tela, hay que revisar dimensiones físicas, resolución de impresión, márgenes de repetición, perfil de color y requisitos del proveedor.

## Pruebas

Con las dependencias instaladas:

    python -m unittest discover -s tests -v

Las pruebas verifican el catálogo, reproducibilidad, dimensiones, generación de los patrones registrados, mosaico y metadatos de exportación.

## Archivos principales

| Archivo | Función |
| --- | --- |
| **pattern_library.py** | Biblioteca original de patrones; se reutiliza como fuente de generación. |
| **pattern_lab.py** | Validación, generación reproducible, vista repetida y metadatos. |
| **app.py** | Servidor web local basado en la biblioteca estándar de Python. |
| **templates/index.html** | Interfaz y maquetas vectoriales 2D. |
| **tests/test_pattern_lab.py** | Pruebas de las funciones principales de Pattern Lab. |
| **requirements-pattern-lab.txt** | Dependencias mínimas del explorador textil. |

Los módulos heredados **eval_core.py**, **eim_detector.py** y **textMode.py** se conservan sin cambios en esta etapa.

## Evaluación con detector — fuera del alcance de esta versión

El proyecto original también contiene una etapa experimental para aplicar patrones a una fotografía usando una máscara de prenda y medir la respuesta de un modelo Edge Impulse local. Esa etapa no se ejecuta desde el nuevo explorador web.

El módulo heredado **eim_detector.py** usa un archivo .eim mediante el SDK de Edge Impulse y documenta requisitos compatibles con Linux, WSL2 o un contenedor Linux. Para esa función se necesitan sus dependencias propias; **requirements-pattern-lab.txt** instala solamente lo que necesita el explorador de patrones.

Una etapa futura puede reutilizar las semillas y los metadatos exportados para comparar imágenes de forma controlada, pero una vista de prenda no demuestra por sí sola que un patrón afecte el reconocimiento. Eso requerirá pruebas con detectores reales y condiciones comparables.

## Licencia y atribución

Este trabajo deriva de [ronibandini/AdversarialPatternsLab](https://github.com/ronibandini/AdversarialPatternsLab). Conservá el archivo LICENSE, los avisos de autoría y las condiciones aplicables de GPL-3.0 al redistribuir el proyecto o versiones modificadas.
