# Entorno Python de ModelBuilder

Fecha: 2026-11-08. Estado: **instalado y validado en Linux x86_64**.

## Entorno único

La aplicación usa `/home/randy/Python/ModelBuilder/.venv/bin/python`, creado con `/usr/bin/python3.12` (Python 3.12.3) y `include-system-site-packages=false`. El entorno contiene entrenamiento existente, importación de datasets, EG2, vídeo y YOLO. La Conda base `/home/randy/miniconda3` no se modificó. El venv inicial que solo tenía pip se conserva en `workspace_data/.venv-modelbuilder-incomplete/`; no se debe reutilizar porque fue movido y sus shebangs apuntan a la ruta vieja.

## Versiones instaladas

- PyTorch 2.12.0+cu130 y torchvision 0.27.0+cu130; CUDA disponible en la NVIDIA RTX 3060.
- Transformers 5.22.2 y Sentence Transformers 6.2.0.
- NumPy 1.26.4, Pillow 13.0.0, Hugging Face Hub 1.33.0, datasets 4.8.5, KaggleHub 1.0.3.
- Accelerate 1.15.0, PEFT 0.21.2, safetensors 0.8.0, PyArrow 21.0.0.
- Ultralytics 8.4.208, OpenCV 4.11.0.86 y pytest 9.2.0.

`pyproject.toml` fija las versiones críticas del modelo y el par Torch/torchvision; `constraints-linux-py312-cu130.txt` congela las 103 distribuciones exactas verificadas para CPython 3.12 en Linux CUDA 13.0. Ese lock es específico de plataforma. `scripts/install-linux.sh` lo aplica cuando detecta Python 3.12 y una GPU NVIDIA; en otras variantes instala los requisitos declarados en `pyproject.toml`. `scripts/install-windows.ps1` selecciona Python 3.12/3.11, usa la rueda CPU o CUDA detectada y ejecuta `pip check`; no se pudo ejecutar PowerShell desde este host Linux.

`pip check` en `.venv` terminó con `No broken requirements found`. Los conflictos de click y ml-dtypes detectados originalmente en Conda pertenecen a esa base y no afectan este entorno aislado.

## Compatibilidad de EmbeddingGemma 2

La configuración oficial de `google/embeddinggemma-2` declara `model_type: embedding_gemma2` y `transformers_version: 5.18.0.dev0`. Transformers 5.10.2 no reconocía el tipo. La arquitectura ya está en la documentación oficial de Transformers; el release estable 5.22.2 la carga. La documentación recomienda Sentence Transformers 6.1 o posterior; se instaló la estable 6.2.0. ModelBuilder utiliza directamente `AutoProcessor` y `AutoModel`, sin `trust_remote_code` ni Python descargado del Hub.

Fuentes primarias consultadas:

- [Configuración oficial del modelo](https://huggingface.co/google/embeddinggemma-2/blob/main/config.json)
- [Documentación oficial EmbeddingGemma2 de Transformers](https://huggingface.co/docs/transformers/v5.22.2/model_doc/embedding_gemma2)
- [Código y guía oficial del modelo en Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/embedding_gemma2.md)
- [Transformers 5.22.2 en PyPI](https://pypi.org/project/transformers/5.22.2/)
- [Sentence Transformers 6.2.0 en PyPI](https://pypi.org/project/sentence-transformers/6.2.0/)
- [Matriz oficial PyTorch/torchvision](https://pytorch.org/get-started/previous-versions/)

## Verificaciones completadas

- Importaron Transformers, Sentence Transformers, Torch/vision, datasets, Hugging Face Hub, safetensors, Accelerate, PEFT, PyArrow, KaggleHub, Ultralytics, CV2 y `modelbuilder.engine`, `dataset_models` y `dataset_project`.
- `torch.cuda.is_available()` devolvió `True`, dispositivo RTX 3060, build CUDA 13.0.
- En la revisión `c3f2237ce16fe33d1bb9dab8692c29c79865bdeb`, `AutoConfig` produjo `EmbeddingGemma2Config` y `AutoModel` resolvió `EmbeddingGemma2Model`. `AutoProcessor` procesó una imagen (`pixel_values (1, 2520, 768)`) y texto (`input_ids (1, 11)`) sin bajar pesos.
- OpenCV codificó y decodificó un MP4 sintético con 4 de 4 fotogramas (24×32).
- Suite completa backend: `179 passed in 15.27s` con `.venv/bin/python -m pytest -q -p no:cacheprovider -o addopts=''`.
- Inferencia EG2 real a través de `modelbuilder.dataset_models.run_embeddings`: clasificación de una imagen 48×32 frente a dos textos en CPU `float32`. Revisión fijada `c3f2237ce16fe33d1bb9dab8692c29c79865bdeb`; job completado. Vector de imagen: dimensión 768, finito, norma L2 `0.99999994`. La codificación de texto pasó la misma comprobación interna de finitud/norma y coincidió en dimensión 768; similitudes `0.72348058` y `0.64191198`.
- El implementador backend repitió la clasificación real después de usar `dtype=torch.float32` (API actual de Transformers 5.22.2) y validó clustering real con dos imágenes: dos grupos/propuestas, dimensión 768.

El shard descargado mide 1,488,915,288 bytes. Se guardó en el caché predeterminado de Hugging Face (`~/.cache/huggingface/hub/models--google--embeddinggemma-2`) que usa la aplicación, no dentro del repositorio. No se solicitó ni descargó otro checkpoint para esta prueba. Tras la descarga quedaban 13 GB libres. La prueba de configuración previa dejó un caché pequeño de tokenizer/config en `workspace_data/.hf-cache`; no contiene pesos y no se selecciona al ejecutar la aplicación.

## Instalación futura

En Linux, ejecutar `scripts/install-linux.sh`. Para recrear exactamente el entorno probado en Linux CUDA, crear `.venv`, instalar Torch desde el índice oficial cu130 y aplicar el lock:

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --timeout 180 --retries 8 \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  'torch==2.12.0' 'torchvision==0.27.0' 'numpy>=1.26,<2.0' 'Pillow>=10,<15'
.venv/bin/python -m pip install --no-cache-dir -c constraints-linux-py312-cu130.txt -e '.[dev]'
.venv/bin/python -m pip check
```
