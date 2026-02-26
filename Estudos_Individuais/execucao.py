# =========================
# 1) RUNNER (orquestrador) - run_all.py
# =========================
import importlib.util


def run_script(script_path, input_file, credentials_json, spreadsheet_id, data_ref):
    print(data_ref)
    # Carregar o script como um módulo
    spec = importlib.util.spec_from_file_location("etl_module", script_path)
    module = importlib.util.module_from_spec(spec)

    # Definir variáveis globais dentro do módulo (injeção)
    module.input_file = input_file
    module.credentials_json = credentials_json          # caminho do JSON do service account
    module.spreadsheet_id = spreadsheet_id
    module.data_ref = data_ref 
    print(module.data_ref)                         # <-- data passada por parâmetro

    # Executar o módulo (isso roda o código "automático" do script)
    spec.loader.exec_module(module)


if __name__ == "__main__":
    scripts_to_run = [
        # "./Apartamento_Asa_Sul.py",
        # "./Apartamento_Asa_Norte.py",
        # "./Apartamento_Noroeste.py",
        # "./Apartamento_Aguas_Claras.py",
        # "./Apartamento_Guara.py",
        # "./Apartamento_Park_Sul.py",
        # "./Apartamento_Sudoeste.py",
        # "./Casa_Arniqueira.py",
        # "./Casa_Asa_Sul.py",
        # "./Casa_Guara.py",
        # "./Casa_Jardim_Botanico.py",
        "./Casa_Lago_Norte.py",
        "./Casa_Lago_Sul.py"
        # "./Casa_Park_Way.py",
        # "./Casa_Sobradinho(Alto da boa vista).py",
        # "./Casa_Sobradinho.py",
        # "./Casa_Vicente_Pires.py"
    ]

    input_file = "./2026-01-18.csv"
    credentials_json = "./credentiasl_machome.json"
    spreadsheet_id = "1oyKvXEcOe5jJITxH4EHe4xOmVp-uT_5XIRvuXQe-Esw"
    # spreadsheet_id = "1rBJ2XlS8b5ynu9AS28sgYDcAdiGyChrJ_6P-tJwg3BM" # COPIA
    data_ref = '2026-01-18'

    for script in scripts_to_run:
        print(f"Running script: {script}")
        run_script(script, input_file, credentials_json, spreadsheet_id, data_ref)
        print(f"Finished running script: {script}")
