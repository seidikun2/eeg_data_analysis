"""Interface local: streamlit run app.py."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from eeg import BLOCKS, PROTOCOLS, save_table, scan, selection, settings, truth

st.set_page_config(page_title="EEG · relatórios", layout="wide")
st.title("EEG · análise e relatórios")
st.caption("Dados locais · controle manual · análise eletrofisiológica e classificação por trial")
root_text = st.sidebar.text_input("Pasta dos dados", value="")
out_text  = st.sidebar.text_input("Pasta de saída", value="analysis_outputs")
cfg_text  = st.sidebar.text_input("Configuração JSON (opcional)", value="")
if not root_text:
    st.info("Informe a pasta que contém os CSVs signal/markers. Subpastas são pesquisadas automaticamente.")
    st.stop()
root    = Path(root_text).expanduser().resolve()
control = root / "dataset_index.csv"
if not root.is_dir():
    st.error("Pasta inexistente.")
    st.stop()
if st.sidebar.button("Atualizar índice", help="Preserva decisões existentes; novos arquivos começam excluídos."):
    try:
        scan(root, control)
        st.success("Índice atualizado.")
    except Exception as exc:
        st.error(str(exc))
if not control.exists():
    st.info("Clique em Atualizar índice para criar a tabela de controle.")
    st.stop()
frame                                  = pd.read_csv(control, keep_default_na=False)
frame["include"]                       = truth(frame.include)
for column in ["bad_channels", "exclude_trials", "reason", "train_run_id"]:
    if column not in frame:
        frame[column] = ""
subjects                               = st.sidebar.multiselect("Sujeitos · vazio = todos", sorted(frame.subject.unique()))
subset                                 = frame[frame.subject.isin(subjects)] if subjects else frame
sessions                               = st.sidebar.multiselect("Sessões · vazio = todas", sorted(subset.session.unique()))
blocks                                 = st.sidebar.multiselect("Tipos de bloco · vazio = todos", BLOCKS)
runs                                   = st.sidebar.multiselect("Registros · vazio = todos", subset.run_id.tolist())
tab_control, tab_analysis, tab_results = st.tabs(["Controle dos dados", "Produzir relatório", "Consultar resultados"])
with tab_control:
    st.write("Edite inclusão, tipo de bloco, grupo, canais ruins e exclusões. IDs de trials começam em zero.")
    editable = ["include", "group", "block_type", "bad_channels", "exclude_trials", "reason", "train_run_id"]
    editor   = st.data_editor(subset, hide_index=True, disabled=[c for c in frame if c not in editable],
                             column_config={"block_type": st.column_config.SelectboxColumn(options=BLOCKS + ["online_definir"]),
                                            "exclude_trials": st.column_config.TextColumn(help="IDs separados por vírgula: 0,3,8"),
                                            "bad_channels": st.column_config.TextColumn(help="Nomes separados por vírgula: C3,CPz"),
                                            "train_run_id": st.column_config.TextColumn(help="Opcional: IM_treino anterior do mesmo sujeito para avaliação online proxy.")},
                             key=f"editor_{root}")
    if st.button("Salvar controle", type="primary"):
        saved = frame.set_index("run_id")
        for row in editor.itertuples(index=False):
            for col in editable:
                saved.loc[row.run_id, col] = getattr(row, col)
        save_table(saved.reset_index(), control)
        st.success("Decisões salvas em dataset_index.csv.")
with tab_analysis:
    st.write("Os filtros da barra lateral são aplicados aos registros incluídos na tabela salva.")
    mode      = st.selectbox("Análises", ["both", "ephys", "classification"],
                        format_func=lambda s: {"both": "Eletrofisiologia + classificação", "ephys": "Eletrofisiologia", "classification": "Classificação"}[s])
    detailed  = st.checkbox("Relatório minucioso: cada época antes/depois", value=True)
    protocols = st.multiselect("Protocolos", list(PROTOCOLS), default=["left_right"])
    try:
        cfg = settings(cfg_text or None)
        with st.expander("Parâmetros efetivos"):
            st.json(cfg)
        chosen = selection(control, subjects, sessions, blocks, runs)
        st.write(f"{len(chosen)} registros selecionados.")
        if not chosen.empty:
            st.dataframe(chosen[["run_id", "subject", "session", "block_type"]], hide_index=True)
        if st.button("Gerar relatórios", type="primary", disabled=chosen.empty or (mode != "ephys" and not protocols)):
            from analyze import batch
            output = Path(out_text).expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            with st.spinner("Processando. Busca aninhada pode demorar; resultados serão gravados por registro."):
                results = batch(root, control, output, cfg, subjects, sessions, blocks, runs, mode, detailed, protocols)
            st.session_state["last_output"] = str(output)
            st.success(f"Execução concluída: {output}")
            st.json(results)
    except Exception as exc:
        st.error(str(exc))
with tab_results:
    output_root = Path(out_text).expanduser()
    choices     = sorted([p.parent for p in output_root.glob("*/batch_results.json")], reverse=True) if output_root.exists() else []
    if not choices:
        st.info("Gere um relatório para consultar os resultados aqui.")
    else:
        output  = st.selectbox("Execução", choices, format_func=lambda p: p.name)
        results = json.loads((output / "batch_results.json").read_text())
        visible = [r for r in results if (not subjects or r["subject"] in subjects)
                   and (not sessions or r["session"] in sessions) and (not blocks or r["block_type"] in blocks)
                   and (not runs or r["run_id"] in runs)]
        for result in visible:
            with st.expander(result["run_id"], expanded=len(visible) == 1):
                if result["status"] != "ok":
                    st.error(result["status"])
                    continue
                folder = output / result["run_id"]
                st.image(str(folder / "dashboard.png"))
                st.download_button("Baixar relatório HTML completo", (folder / "report.html").read_bytes(),
                                   file_name=f"{result['run_id']}.html", mime="text/html", key=result["run_id"])
                figures = sorted(folder.glob("*.png"))
                figure  = st.selectbox("Inspecionar figura", figures, format_func=lambda p: p.stem, key=f"fig_{result['run_id']}")
                st.image(str(figure))
                st.json(result.get("classification", []))
