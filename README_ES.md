# BasinCast API v1.0 - Instalacion y despliegue gratuito

## Que contiene

Este proyecto incorpora **sin cambios** el motor cientifico `basincast/` de BasinCast Engine v1.0 y una capa HTTP de FastAPI en `api/main.py`. No depende de Gemini, ni de una base de datos, ni de meteorologia para pronosticar series aportadas por los usuarios. El pronostico de una serie de caudal no es una prediccion de demanda de agua.

- `GET /health`: estado y version del motor.
- `GET /v1/info`: contrato y limites de la API.
- `POST /v1/forecast`: recibe CSV/XLSX y devuelve un JSON con pronosticos y metricas.
- `POST /v1/forecast.zip`: recibe los mismos campos y devuelve el ZIP cientifico completo.
- `GET /docs`: formulario interactivo para probar la API.

El CSV minimo debe contener `date,value` con fechas de calendario e intervalo mensual. Si hay varias muestras por mes, hay que indicar agregacion (`mean`, `sum` o `last`), de acuerdo con el significado fisico de la variable. No se interpolan los huecos del objetivo.

## Ejecucion local: totalmente gratuita

Requiere Python 3.10 o posterior (recomendado Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

En Windows, activar con `.venv\\Scripts\\activate`.

Abrir http://127.0.0.1:8000/docs y probar `POST /v1/forecast` con `examples/river_2_columns.csv`. Campos: `series_id=River_8107`, `asset_type=river`, `unit=hm3/month`, `max_horizon_months=3`.

Para guardar ZIP desde una terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/forecast.zip \
  -F 'file=@examples/river_2_columns.csv' \
  -F 'series_id=River_8107' \
  -F 'asset_type=river' \
  -F 'unit=hm3/month' \
  -F 'max_horizon_months=3' \
  -o basincast_forecast.zip
```

Para ejecutar pruebas:

```bash
python -m pytest -q tests/test_api.py tests/test_engine_v1.py
```

## Despliegue gratuito de demostracion en Render

1. Crear una cuenta gratuita en GitHub y un repositorio privado. Subir el **contenido** de esta carpeta al directorio raiz del repositorio. No subir claves API, historicos privados ni ZIP de datos de usuarios.
2. Crear cuenta gratuita en https://render.com/ sin metodo de pago. En Dashboard seleccionar **New > Web Service**, conectar el repositorio y elegir **Free**. Alternativamente, usar el `render.yaml` incluido (Blueprint).
3. Configurar compilacion: `pip install -r requirements-api.txt`; inicio: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`; version de Python: `3.12`.
4. Configurar variables de entorno **en el panel de Render**, no en GitHub:
   - `BASINCAST_REQUIRE_API_KEY=1`
   - `BASINCAST_API_KEY=<un_secreto_largo_aleatorio>` (generar una clave distinta para cada despliegue)
   - `BASINCAST_MAX_UPLOAD_BYTES=5242880`
   - `BASINCAST_MAX_CONCURRENT_JOBS=1`
5. Verificar `https://<tu-servicio>.onrender.com/health`. La pagina `/docs` permite hacer pruebas introduciendo `X-API-Key` en las solicitudes; si usas un cliente HTTP, enviar esa cabecera.

Render Free no tiene SLA, se duerme despues de 15 minutos sin solicitudes, puede tardar alrededor de un minuto en despertar, no dispone de disco persistente y tiene cupos de CPU/transferencia. **No introducir un metodo de pago** si se requiere coste monetario estrictamente cero: cuando se agotan cupos, el servicio puede suspenderse. Ver https://render.com/docs/free. El plan gratuito no es adecuado para una aplicacion publica de produccion.

## Conexion desde Google AI Studio

1. Crear una web app en Build Mode. Su servidor Node/Express, **no el navegador**, debe llamar a FastAPI.
2. Configurar secretos de servidor en AI Studio (Settings > Secrets):
   - `BASINCAST_API_URL=https://<tu-servicio>.onrender.com`
   - `BASINCAST_API_KEY=<el_mismo_secreto_que_en_Render>`
3. Pegar el prompt contenido en `AI_STUDIO_PROMPT_EN.md`. No utilizar Gemini para calcular pronosticos. No poner API KEY ni URL de la API dentro del codigo React cliente: el backend Node leera los valores de `process.env`.
4. Para publicar sin costes de hosting, seleccionar Google Cloud Starter Tier **solo si aparece disponible en esa cuenta**. Este nivel permite hasta dos apps y no requiere facturacion; cuentas de Workspace o con historial de facturacion pueden no cumplir los requisitos.

## Interpretacion cientifica

- `forecast` contiene **todos** los candidatos calculados (hasta 48 meses): no implica que todos sean fiables.
- `recommended_forecast` contiene unicamente los meses pertenecientes a un tramo continuo respaldado por las reglas de KGE, soporte temporal y validacion.
- `metrics` informa KGE en validacion y test, RMSE y mejora de Ridge frente a la referencia como **indicadores diferentes**.
- Series cortas pueden producir una prediccion estacional no validada; no se infiere KGE de esos pronosticos.
- Los resultados de un nivel de acuifero o un caudal NO cuantifican demanda de agua si no se aporta una serie de consumo o un modelo fisico adicional.
- Escenarios de cambios IPCC: fuera de esta API minima. Una perturbacion determinista de la climatologia historica **no** transforma por si sola los niveles o caudales futuros.

## Seguridad, datos y costes

La API no persiste las cargas; cada peticion utiliza un directorio temporal que se borra al finalizar. Se limita el tamano del archivo (5 MiB), los registros (6000), el contenido descomprimido de XLSX (40 MiB), la concurrencia (1) y los horizontes (48). El codigo calcula sin registrar el historico en logs. Para datos sensibles o una publicacion abierta, revisar autenticacion por usuario, proteccion contra abuso, tratamiento de datos personales y limites de solicitudes antes de habilitar un acceso amplio. La clave compartida no es una solucion completa de identidad de usuarios.

Este paquete **no incorpora** APIs de pago, bases de datos ni llamadas externas necesarias para generar los pronosticos.
