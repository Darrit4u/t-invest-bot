from t_tech.invest import Client, InstrumentStatus
import os

from dotenv import load_dotenv
load_dotenv()


TOKEN = os.environ["INVEST_TOKEN"]

# получить инфу о тикерах для DEFAULT_INSTRUMENTS
with Client(TOKEN) as client:
    resp = client.instruments.futures(
        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
    )

    for f in resp.instruments:
        print(
            f"name={f.name} | ticker={f.ticker} | class_code={f.class_code} | "
            f"figi={f.figi} | uid={f.uid}"
        )