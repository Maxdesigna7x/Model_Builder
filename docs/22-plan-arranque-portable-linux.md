# Plan de implementación: arranque fiable de escritorio en Linux

Estado: implementación parcial. Ya existen `--graphics=compatible`, `--diagnose` y el mensaje HTML inicial. La auditoría de paquetes, registros tempranos, recuperación gráfica y pruebas de arranque de los instaladores siguen pendientes. Este documento conserva el alcance completo del arreglo.

## Objetivo y alcance

Al abrir ModelBuilder por primera vez en un Linux compatible, debe aparecer la interfaz de preparación sin instalar Python, Node, Rust o PyTorch manualmente. El paquete base debe contener los recursos necesarios para mostrar esa interfaz. Después, con conexión, se descargará el motor elegido: CPU o NVIDIA.

Objetivo inicial verificable: Linux x86_64, Zorin basado en Ubuntu 22.04 o posterior. Registrar las versiones concretas probadas; no prometer compatibilidad con cualquier Linux. Conservar el modo web y Windows. No migrar a Electron ni reescribir React/PyTorch.

## Situación comprobada y límites del diagnóstico

- La app abre en el equipo del desarrollador y en Windows. En otro Zorin, tanto AppImage como DEB abren una ventana vacía.
- No tenemos todavía los registros ni la versión exacta/controlador/sesión gráfica de ese equipo. La causa sigue sin determinarse.
- Linux usa WebKitGTK. Hay problemas conocidos de renderizado con ciertas combinaciones de controladores, NVIDIA y Wayland/X11. También pueden existir bibliotecas ausentes o errores de JavaScript.
- `src-tauri/src/main.rs` llama directamente a `modelbuilder_lib::run()`; no hay configuración gráfica de compatibilidad ni diagnóstico temprano.
- `frontend/index.html` contiene un `root` vacío. Si React no arranca, no ofrece ninguna explicación.
- `frontend/src/DesktopSetup.tsx` ofrece CPU/NVIDIA y llama a `runtime_status` y `prepare_runtime`. No puede solucionar un fallo del propio WebView.
- Python se descarga ya mediante uv. No es necesario modificar el motor de entrenamiento para corregir la ventana inicial.
- El workflow actual compila y comprueba Python, pero no prueba el arranque gráfico de los paquetes en una instalación limpia.

## Decisiones que debe respetar la implementación

1. Mantener un paquete base + descarga posterior del motor pesado.
2. Separar la aceleración de la interfaz de la aceleración del entrenamiento: abrir la interfaz en modo gráfico conservador no debe desactivar CUDA en PyTorch.
3. No copiar controladores del desarrollador, kernel o glibc al instalador. No modificar controladores, permisos de dispositivos ni ajustes globales del sistema.
4. En DEB, las bibliotecas nativas del sistema se declaran como dependencias y las instala el gestor de paquetes. Un DEB no será anunciado como autónomo/offline si necesita descargarlas.
5. En AppImage, incluir las bibliotecas redistribuibles necesarias mediante el empaquetado soportado por Tauri/linuxdeploy. Su autonomía debe verificarse, no deducirse de la extensión.
6. No descargar bibliotecas nativas arbitrarias en el primer inicio, ni mezclar WebKitGTK de una distribución con glibc de otra. Si el empaquetado actual no permite una base coherente, documentar el bloqueo antes de añadir copias manuales o cambiar de tecnología.
7. Nunca borrar `workspace_data` ni el entorno privado durante un intento de recuperación gráfica.
8. Respetar las preferencias del usuario: pruebas breves de arranque; no esperar descargas completas de CUDA ni compilaciones remotas salvo petición explícita.

## Orden de trabajo

### 1. Recoger evidencia y auditar el paquete base

Archivos: nuevos `scripts/audit-linux-package.sh` y documentación en `docs/INSTALACION.md`.

- Registrar distribución/versión, arquitectura, sesión X11/Wayland, versión del paquete, GPU y controlador cuando puedan detectarse. La ausencia de `nvidia-smi` no prueba ausencia de NVIDIA.
- Extraer DEB y AppImage en carpetas temporales. Para AppImage, contemplar extracción y ejecución sin FUSE además de ejecución normal.
- Inspeccionar con `readelf`/`ldd` el ejecutable, uv y los componentes/helper processes de WebKit incluidos. No basta con revisar el binario principal: existen cargas dinámicas.
- Inspeccionar `dpkg-deb --info` y `Depends`; detectar bibliotecas sin resolver en un entorno limpio objetivo, no solo en el equipo de desarrollo.
- Generar un informe: biblioteca requerida, origen (incluida/sistema), paquete del sistema cuando corresponda y resultado de comprobación.
- Comprobar el baseline ABI del ejecutable y de uv. Mantener la compilación Linux en Ubuntu 22.04 o el baseline más antiguo expresamente soportado.
- Comparar el informe con el equipo afectado si sus registros están disponibles. Si no están, continuar con las mejoras generales y dejar explícito que el caso del hermano no está confirmado como resuelto.

### 2. Añadir diagnóstico antes de crear el WebView

Archivos: nuevo `src-tauri/src/startup.rs`; integración al principio de `src-tauri/src/main.rs` y `lib.rs`.

- Inicializar un registro acotado de arranque en la carpeta de datos del usuario, antes de crear la ventana.
- Registrar versión de app, modo gráfico, hitos `process_started`, `webview_created`, `frontend_ready`, `setup_visible`/`app_visible` y fallos.
- Conservar algunos registros recientes y limitar tamaño. No registrar credenciales, contenido de datasets ni entorno completo.
- Instalar un manejador de panic que escriba el error antes de finalizar.
- Añadir `--diagnose` para obtener un informe breve sin descargar Python/PyTorch. Si no hay servidor gráfico, debe terminar con un mensaje legible por terminal.
- Límite importante: si el cargador dinámico falla antes de ejecutar Rust, el registro interno no puede funcionar. Cubrir ese caso con la auditoría y el informe del lanzador de AppImage, no con una promesa de recuperación dentro de React.

### 3. Añadir modo gráfico de compatibilidad

Archivos: `startup.rs`, `main.rs`, integración de reinicio en `lib.rs`; lanzador/entrada de escritorio del paquete según el resultado de la auditoría.

- Admitir `--graphics=auto` (predeterminado) y `--graphics=compatible`.
- En modo compatible de Linux, establecer `WEBKIT_DISABLE_DMABUF_RENDERER=1` antes de crear WebKit y antes de arrancar hilos.
- Reservar `WEBKIT_DISABLE_COMPOSITING_MODE=1` como segundo nivel de recuperación, con registro de su activación; probarlo antes de distribuirlo.
- No imponer estos ajustes a todos los Linux sin evidencia. Respetar las variables ya establecidas por el usuario y registrar el modo efectivo.
- Para NVIDIA detectada, usar el ajuste mínimo comprobado en el equipo afectado. Si no se ha confirmado esa detección/solución, ofrecer el modo compatible explícito en vez de adivinar.
- Crear una entrada de escritorio/acción «Abrir en modo compatible» que ejecute el mismo programa con el argumento correspondiente. Debe poder usarse aunque la ventana principal no muestre nada.
- Si se permite reiniciar desde un mensaje de recuperación, limitarlo a un intento por nivel y no crear un bucle. No utilizar el estado «la última ejecución no terminó» como única prueba de fallo gráfico.
- Este modo afecta únicamente a la interfaz. Verificar que no se cambie `CUDA_VISIBLE_DEVICES`, la selección CPU/NVIDIA ni la configuración del entrenamiento.

### 4. Evitar el silencio cuando falla el frontend

Archivos: `frontend/index.html`, `frontend/src/main.tsx`, nuevo componente de error; nuevos comandos/eventos y permisos concretos en `src-tauri/capabilities/default.json`.

- Añadir al HTML base un mensaje visible «Abriendo ModelBuilder…» con estilos locales mínimos. React lo sustituye al montar correctamente. No cargar recursos remotos.
- Incorporar un límite de errores de React y registrar errores de carga de recursos/JavaScript; ofrecer un diagnóstico legible en lugar de dejar `root` vacío.
- Enviar un acuse explícito al backend cuando el frontend haya montado y cuando se muestre preparación o app. Hacerlo independiente de la descarga de Python.
- Implementar un temporizador de arranque (por ejemplo, 20 segundos sin acuse, configurable en pruebas). No usarlo para el tiempo de descarga de dependencias.
- Ante falta de acuse, guardar el informe e intentar un aviso nativo que no dependa de WebKit, usando una ruta disponible y verificada en Linux (evaluar el `rfd::MessageDialog` ya incluido). El aviso debe ofrecer modo compatible y ubicación del registro.
- No asumir que un aviso nativo puede mostrarse si faltan las bibliotecas base de su propio toolkit. En ese caso, conservar salida a terminal y diagnóstico de empaquetado.
- Un acuse JavaScript NO demuestra que los píxeles se estén dibujando. Por eso debe seguir existiendo el acceso externo al modo compatible y una comprobación visual real.

### 5. Corregir el empaquetado según la auditoría

Archivos: `src-tauri/tauri.conf.json`, `.github/workflows/desktop.yml`, scripts de empaquetado que sean necesarios, `docs/INSTALACION.md`.

- Conservar solo código del backend y recursos de arranque; excluir proyectos, cachés y entornos del desarrollador.
- DEB: verificar y corregir las dependencias de runtime que no detecte Tauri automáticamente. Determinar los nombres exactos con la auditoría, no copiarlos de una distribución distinta.
- AppImage: verificar que la cadena de WebKitGTK y sus helpers esté incluida o resuelta de forma soportada. No inventar `LD_LIBRARY_PATH` globales ni empaquetar drivers para esconder errores.
- Conservar uv incluido y su permiso de ejecución. La primera pantalla debe abrir sin Python instalado y sin conexión; la instalación del motor sí requiere conexión.
- Windows: mantener el comportamiento que funciona. Documentar cómo el instalador proporciona WebView2 si falta; no aplicar variables WebKit de Linux en Windows.
- Documentar qué incluye cada formato y qué sigue dependiendo del sistema: gestor de paquetes, sesión gráfica, controladores y, si corresponde, FUSE.

### 6. Verificación breve antes de publicar

Archivos: nuevo `scripts/test-desktop-startup.*`, extensión del workflow y guía de prueba manual.

- Probar el paquete release real, no solamente Vite ni el backend Python.
- En Linux limpio objetivo, sin Python de desarrollo, abrir DEB instalado y AppImage con red desactivada. Debe verse la preparación/selector sin descargar nada ni tocar proyectos del desarrollador.
- Probar arranque normal y compatible; probar AppImage con y sin FUSE cuando corresponda.
- En CI, usar una sesión gráfica de prueba y un directorio de usuario temporal. Verificar los hitos, la permanencia del proceso, la ausencia de fallos y capturar una imagen. Xvfb es una comprobación adicional: no representa NVIDIA/Wayland reales.
- Añadir pruebas de frontend para error de montaje, falta de respuesta de `runtime_status`, error de carga y recuperación. Verificar que la espera de Python no active el temporizador gráfico.
- Verificar modo web y arranque de Windows como regresiones.
- Terminar después de comprobar la primera pantalla y el inicio de preparación. No realizar la descarga completa del motor por defecto para esta tarea.
- Probar físicamente el nuevo paquete en el Zorin afectado con el mismo modo de escritorio/driver. Si no se tiene acceso, declarar esa prueba pendiente.
- Incorporar el smoke test de arranque al job Linux antes de subir sus artefactos. Si falla, no publicar automáticamente una nueva Release.

## Criterios de aceptación

- [ ] En un Zorin limpio y expresamente soportado se abre la primera pantalla sin Python/Node/Rust instalados y sin Internet.
- [ ] Se documenta y comprueba qué bibliotecas incluye AppImage y cuáles instala el gestor para DEB.
- [ ] Existen registros tempranos acotados y un informe útil sin descargar el motor.
- [ ] Se puede iniciar en modo compatible desde fuera de la ventana principal.
- [ ] El modo compatible de interfaz no cambia la selección NVIDIA ni desactiva CUDA.
- [ ] Un error de React o falta de acuse ofrece diagnóstico; no se confunde la espera de una descarga con un fallo de arranque.
- [ ] No se altera ningún proyecto ni el Python del sistema durante recuperación.
- [ ] Las comprobaciones gráficas del paquete son previas al job de publicación.
- [ ] Windows y web conservan su funcionamiento.
- [ ] Se indica por separado si se confirmó la solución en el equipo del hermano. No declarar resuelto ese caso solo porque CI pasa.

## Entrega del agente implementador

1. Implementar y registrar cambios, pruebas breves y límites conocidos.
2. Actualizar instalación y `AGENTS.md` si cambia el comando de compilación o diagnóstico.
3. Cuando esté autorizado generar/publicar, subir el commit e iniciar `gh workflow run desktop.yml --ref main`. Este flujo publica automáticamente si los jobs pasan.
4. Respetar la regla existente: no esperar ni consultar repetidamente las compilaciones remotas salvo solicitud explícita del usuario. Informar el lanzamiento sin afirmar que terminó.

## Referencias primarias

- Problemas gráficos y ajustes de WebKitGTK: https://v2.tauri.app/develop/debug/linux-graphics/
- Empaquetado AppImage y baseline del sistema: https://v2.tauri.app/distribute/appimage/
- Dependencias de paquetes Debian: https://v2.tauri.app/distribute/debian/

Consultar las versiones de documentación compatibles con las herramientas fijadas en el repositorio antes de implementar los detalles de empaquetado.
