# Módulo 2 — Estimación de peso bovino

Prueba de concepto del pipeline de inferencia de peso, validada sobre un dataset
público con peso de báscula antes de salir a recolectar datos propios.

## Enfoque

**Sin escala métrica.** No se mide en centímetros ni se usa marcador, giroscopio ni
LiDAR. El peso se infiere de la **forma** de la silueta, aprovechando que un bovino
no escala uniformemente: un ternero de 200 kg tiene proporciones distintas a una vaca
de 500 kg. Es lo que hacen los productos comerciales que ya funcionan en el mercado
(Peso Certo en Brasil reporta ~8,6% de error por animal con una sola foto lateral).

El objetivo del acta es MAPE < 15%. El techo conocido del 2D es ~8,6%.

## Pipeline

```
imagen → YOLO-seg (clase 'cow' de COCO, sin entrenar) → máscara
       → descriptores invariantes a escala → regresión → kg
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas. El dataset vive en `data/`, ignorado en git (2,29 GB) |
| `src/preparar.py` | Descomprime el zip e inspecciona la estructura real |
| `src/segmentar.py` | YOLO-seg sobre cada imagen → máscara binaria |
| `src/features.py` | 71 descriptores de forma, ninguno en centímetros |
| `src/entrenar.py` | GroupKFold por animal + Ridge/Boosting + MAPE por franja |

## Paso manual: bajar el dataset (cada integrante, una vez)

El dataset **no está en el repo** (pesa 2,3 GB comprimido, GitHub no lo acepta) y
Mendeley está detrás de Cloudflare, así que tampoco se puede automatizar la descarga.
Cada uno lo baja a mano, una sola vez, después de clonar el repo:

1. Abrir https://data.mendeley.com/datasets/h2s22wr5py/2
2. Botón **Download All** (2,29 GB)
3. Guardar como `h2s22wr5py-2.zip` dentro de `modelo-peso/data/`
   (la carpeta `data/` está en `.gitignore`, así que queda local a cada uno)
4. Correr `python src/preparar.py` para descomprimirlo e inspeccionar la estructura

Son 72 bovinos de Mongolia Interior, vista lateral y trasera, fotografiados con
iPhone 13, con keypoints, perímetro torácico, altura a la cruz, largo corporal,
largo de grupa **y peso corporal medido por expertos**. Licencia CC BY 4.0.

## Correr

```bash
python src/preparar.py     # descomprime e inspecciona
python src/segmentar.py    # ~144 imágenes, unos minutos en CPU
python src/features.py     # instantáneo
python src/entrenar.py     # instantáneo
```

## Decisiones de diseño que conviene recordar

**Partición por animal, no por imagen.** `GroupKFold(groups=animal_id)`. Si las fotos
del mismo animal caen en train y test, el modelo lo memoriza y el MAPE reportado es
falso. El estudio de vista dual particiona explícitamente por identidad del animal.

**Las dos orientaciones.** Un animal puede mirar a izquierda o derecha. Ninguna
heurística de canonicalización resultó confiable (probamos comparar el grosor de los
tercios: falla en toros de tren delantero pesado). En vez de adivinar, se emiten las
dos orientaciones de cada silueta y el modelo aprende la invarianza.

**Tres modelos, no uno.** El script compara Schaeffer sobre las medidas reales (piso),
regresión sobre las medidas reales en cm (techo) y regresión sobre la forma sin escala
(lo que nos interesa). Si el último no llega, los otros dos dicen si el problema está
en las features o en el regresor.

## Resultados de la primera corrida (08/09/2026)

Dataset: 72 bovinos adultos de Mongolia Interior, 341–644 kg (media 482, sd 86).
Validación con `GroupKFold` por animal.

| Modelo | Entrada | MAPE | R² |
|---|---|---|---|
| Baseline tonto | nada (predice la mediana) | **14,90%** | −0,03 |
| Schaeffer | medidas reales en cm, sin entrenar | 5,72% | 0,83 |
| Ridge | medidas reales en cm | 4,43% | 0,89 |
| Boosting | medidas reales en cm | **2,59%** | 0,94 |
| Ridge | forma de la silueta, sin escala | 16,89% | **−0,13** |
| Boosting | forma de la silueta, sin escala | 18,20% | **−0,42** |
| Ridge | forma + tamaño aparente en px | 23,95% | −2,56 |

### Tres conclusiones

**1. El objetivo del acta (MAPE < 15%) no discrimina en este rango de pesos.**
Un modelo que ignora la imagen y predice siempre la mediana da 14,90%. Hay que
reportar la *mejora sobre el baseline*, no el MAPE solo, y el dataset propio tiene
que incluir terneros para que la métrica signifique algo.

**2. El enfoque escala-libre por descriptores de silueta falló.** R² negativo:
peor que una constante. No es falta de datos (con 72 animales y 4 medidas reales,
Ridge da 4,43%), ni de segmentación (las máscaras son excelentes).

**3. La causa es la pose.** Diagnóstico:
- El tamaño en píxeles no correlaciona con ninguna medida real (r < 0,25) →
  la distancia de captura **no** fue constante, aunque el paper diga 1 m.
- La silueta no puede recuperar ni la altura a la cruz (R² = −1,37), que es
  puramente geométrica y visible en la foto.
- Las fotos tienen animales con la cabeza arriba y otros pastando, patas abiertas
  o juntas. Cabeza, cuello, patas y cola se mueven independientemente del peso y
  dominan los descriptores de contorno.
- Aislar el torso por erosión morfológica empeora todo (R² −35): es inestable.

### Qué implica

El camino de **keypoints anatómicos no es un refinamiento, es necesario**. Las medidas
que predicen el peso están definidas entre landmarks específicos (cruz, encuentro,
isquion, línea superior e inferior del pecho), y son robustas a la pose de una manera
que el contorno crudo no lo es.

El experimento cuantificó exactamente cuánto cuesta no tenerlos: **17% contra 4%**.

## La escala métrica no hace falta (08/09/2026)

Segunda tanda de experimentos, sobre las medidas reales, con 15 particiones aleatorias
distintas para verificar estabilidad:

| Entrada | MAPE | Estabilidad |
|---|---|---|
| Medidas absolutas en cm | **2,88%** | ± 0,31 |
| **Solo ratios entre medidas (escala-libre)** | **5,53%** | ± 0,48 |
| Control negativo: ratios mezclados al azar | 19,37% | ± 0,97 |
| Baseline tonto | 14,90% | — |

**Los ratios entre landmarks bastan.** Ningún ratio individual correlaciona con el peso
(máximo |r| = 0,26), pero un modelo no lineal sobre el conjunto llega a 5,53%. Es
alometría: los animales grandes no son animales chicos escalados, tienen proporciones
distintas, y esa relación es no lineal (Ridge se queda en 13%, Boosting la captura).

El control negativo se va a 19,37%, peor que el baseline: la señal es real y no un
artefacto del procedimiento de validación.

**Consecuencia para el producto: no hacen falta marcador ArUco, giroscopio, telémetro
ni LiDAR.** Toda la discusión de recuperación de escala se resuelve sola si los
landmarks están bien localizados. Una distancia de captura acotada sigue siendo
deseable por calidad de imagen, pero no por escala.

Advertencia honesta: el 5,53% se midió con medidas *perfectas*, tomadas a mano por
expertos. Los keypoints predichos por un modelo van a tener error y eso degrada el
número. Es el techo del enfoque, no lo que van a obtener. El margen hasta el 15% es
grande, pero hay que medirlo, no suponerlo.

## Estado

- [x] Entorno: torch 2.14 CPU, OpenCV 5, scikit-learn 1.9, ultralytics 8.4
- [x] Pipeline completo corriendo end-to-end
- [x] Segmentación: vista lateral 72/72 (100%), vista trasera 14/72 (19%)
- [x] Baseline, Schaeffer y modelos morfométricos medidos
- [x] Hipótesis escala-libre por silueta probada y **descartada con evidencia**
- [x] Hipótesis escala-libre por ratios entre landmarks **confirmada** (5,53%)
- [x] Proyecto de etiquetado listo: 72 imágenes exportadas + guía de 7 landmarks
- [x] Config y script de YOLO-pose listos
- [ ] Etiquetar las 72 imágenes (18 c/u, ver `etiquetado/README.md`)
- [ ] Entrenar YOLO-pose y medir cuánto degrada el error de keypoint al 5,53%

## Cómo sigue

```bash
python src/exportar_etiquetado.py          # ya corrido: 72 imagenes en etiquetado/imagenes
# etiquetar en CVAT o Roboflow -> exportar formato YOLO pose
python src/entrenar_pose.py                # YOLO-pose, 7 keypoints
python src/medidas_desde_keypoints.py      # keypoints -> ratios -> peso
```
