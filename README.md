# CSLR-IIGA-Refactory: Reconocimiento Continuo de Lengua de Señas con Atención Segmentada y Refactorización Robusta

Repositorio correspondiente al trabajo de titulación: **"Reconocimiento Continuo de Lengua de Señas a partir de YouTube-ASL mediante Segmentación Temporal y Supervisión Lingüística T5"**, desarrollado por **Stephano Marcelo Almendra Chaparro** para optar al título de Ingeniero Civil Informático en la Universidad de Valparaíso (Profesor Guía: Ana Aguilera, Profesor Co-guía: Miguel Yáñez).

---

## Contexto y Propósito del Repositorio (`CSLR-IIGA-Refactory`)

Esta versión refactorizada (**CSLR-IIGA-Refactory**) surge como una evolución crítica del repositorio oficial original de *Intra-Inter Gloss Attention (IIGA)* [3, 19]. El objetivo principal de este código fue auditorar el repositorio oficial, corregir inestabilidades estructurales en el flujo de datos y validar de forma reproducible el modelo sobre el corpus de referencia **PHOENIX-2014** utilizando un clúster local multi-GPU (3x NVIDIA Quadro RTX 4000).

### 🛠️ Principales Mejoras y Refactorizaciones Implementadas
1. **Auditoría y Corrección de Bugs Críticos:**
   - **Persistencia de Máscara:** Corrección del error de sobrescritura en `dataloader.py`.
   - **Consistencia Cromática:** Estandarización estricta al espacio de color RGB mediante OpenCV.
   - **Determinismo de Carga:** Reemplazo de `os.listdir()` por ordenamiento determinista `sorted()` para prevenir desalineaciones temporales entre glosas y secuencias de video.
   - **Resiliencia de Almacenamiento:** Implementación de checkpoints automáticos de seguridad cada 500 pasos (`checkpoint_last_step.pt`) y vaciado físico a disco (`flush()`).

2. **Optimización del Pipeline de Datos y Abstracción Biomecánica:**
   - Reemplazo y supresión de la rama convolucional pesada original (`MobileNetV2`) por extracción directa de landmarks geométricos 2D.
   - Incorporación de **Normalización Centroide** sobre 88 puntos clave de MediaPipe Holistic (pose, manos y rostro), fijando el origen $(0,0)$ en el eje inter-humeral para otorgar invarianza posicional y de escala.

3. **Estabilidad en Entrenamiento Distribuido (DDP):**
   - Integración de un `BucketBatchSampler` consciente de rangos paralelos para maximizar la densidad de lotes sin disparar desbordamientos de memoria (*Out of Memory* en GPUs de 8 GB VRAM).

---

## Estructura de los Archivos Incluidos

- `train.py`: Script principal de entrenamiento adaptado con soporte DDP, control de épocas, autoguardado y pérdida CTC optimizada (`zero_infinity=True`).
- `transformer.py`: Núcleo de la arquitectura de atención IIGA, incluyendo los módulos *Intra-Gloss* (atención local con ventanas de tamaño $w=12$) e *Inter-Gloss* (atención global mediante *Average Pooling*).
- `dataloader.py`: Cargador de datos robusto para PHOENIX-2014 con manejo dinámico de padding, secuencias de video y augmentations.
- `test.py`: Script de evaluación integral sobre conjuntos de validación/prueba con decodificación CTC Beam Search.
- `prediction.py`: Utilidad para inferencia individual sobre clips específicos a partir de anotaciones y checkpoints indexados.
- `generar_graficos_actuales.py`: Script automatizado para la extracción y ploteo de curvas de aprendizaje (`learning_curves.npy`), generando comparativas de WER y pérdida CTC.

---

## Resultados Experimentales en PHOENIX-2014

Tras aplicar la auditoría técnica y la abstracción por landmarks en este repositorio, la superficie de pérdida se simplificó, logrando una convergencia monótona y estable:

- **Reducción del Word Error Rate (WER):** Descenso continuo desde un **70,49%** (baseline inestable sin optimizar) hasta un **48,34%** al cierre de la época 28 de entrenamiento, representando una ganancia neta de **22,15 puntos porcentuales de mejora absoluta**.
- **Perplejidad de Validación:** Reducción del **73,3%** en la perplejidad de validación (de 16,76 a 4,48).
- **Estabilidad frente al Baseline:** Eliminación de los rebotes estocásticos recurrentes del diseño original, demostrando que la abstracción biomecánica es una alternativa viable y computacionalmente eficiente frente a los backbones convolucionales pesados.



---

## Instrucciones de Reproducción (PHOENIX-2014)

Para inicializar el bucle de entrenamiento distribuido multi-GPU sobre PHOENIX-2014:

```bash
torchrun --nproc_per_node=3 train.py \
    --data data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner \
    --lookup_table data/slr_lookup.txt \
    --batch_size 2 \
    --num_epochs 30 \
    --local_window 12
