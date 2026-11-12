"""Interface local: streamlit run app.py."""

from datetime import datetime, timezone
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
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
frame["block_type"]                    = frame.block_type.replace("online_definir", "online_com_feedback")
for column in ["bad_channels", "exclude_trials", "reason", "train_run_id"]:
    if column not in frame:
        frame[column] = ""
subjects                               = st.sidebar.multiselect("Sujeitos · vazio = todos", sorted(frame.subject.unique()))
subset                                 = frame[frame.subject.isin(subjects)] if subjects else frame
sessions                               = st.sidebar.multiselect("Sessões · vazio = todas", sorted(subset.session.unique()))
blocks                                 = st.sidebar.multiselect("Tipos de bloco · vazio = todos", BLOCKS)
runs                                   = st.sidebar.multiselect("Registros · vazio = todos", subset.run_id.tolist())
st.header("1. Selecionar dados e filtros")
# Containers keep the workflow visible and preserve widget state across reruns.
tab_control, tab_analysis, tab_results = st.container(), st.container(), st.container()
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
    st.header("2. Verificar qualidade e 3. Gerar relatório")
    st.write("Os filtros da barra lateral são aplicados aos registros incluídos na tabela salva. O relatório geral reúne todos os registros selecionados.")
    mode      = st.selectbox("Análises", ["both", "ephys", "classification"],
                        format_func=lambda s: {"both": "Eletrofisiologia + classificação", "ephys": "Eletrofisiologia", "classification": "Classificação"}[s])
    detailed = st.checkbox("Incluir inspeção conjunta das épocas antes/depois", value=True)
    protocols = st.multiselect("Protocolos", list(PROTOCOLS), default=["left_right"])
    compare = st.checkbox("Batch comparativo: avaliar TGSP e CSP separadamente", value=False,
                          help="Inclui TGSP e CSP no relatório geral e nas tabelas por sujeito; busca aninhada em cada pipeline. Mais demorado.")
    try:
        cfg = settings(cfg_text or None)
        with st.expander("Ajustar experimento e visualização"):
            cfg["windows"] = st.multiselect("Janelas (s)", sorted(set([.5, 1., 1.5, 2., 2.5, 3.] + cfg["windows"])), default=cfg["windows"])
            cfg["offsets"] = st.multiselect("Offsets de treino (s)", sorted(set([0., .25, .5, 1.] + cfg["offsets"])), default=cfg["offsets"])
            bank_options = {json.dumps(v): v for v in ([[[8,30]], [[5,35]], [[8,13],[13,30]]] + cfg["bands"])}
            banks = st.multiselect("Bandas / bancos a testar", list(bank_options), default=[json.dumps(v) for v in cfg["bands"]])
            cfg["bands"] = [bank_options[v] for v in banks]
            cfg["svm_c"] = st.multiselect("SVM · valores de C", sorted(set([.01,.1,1.,10.,100.] + cfg["svm_c"])), default=cfg["svm_c"])
            cfg["svm_gamma"] = st.multiselect("SVM · gamma", list(dict.fromkeys(["scale", "auto"] + cfg["svm_gamma"])), default=cfg["svm_gamma"])
            cfg["search_budget"] = st.number_input("Máximo de candidatos por busca interna", 1, 200, int(cfg["search_budget"]))
            cfg["curve_smoothing_s"] = st.slider("Suavização visual causal (s)", .1, 2., float(cfg["curve_smoothing_s"]), .1)
            cfg["erd_baseline_mode"] = st.selectbox("Baseline ERD/ERS", ["preceding_pause", "fixed"], index=0 if cfg["erd_baseline_mode"] == "preceding_pause" else 1,
                                                     format_func=lambda v: "PAUSA anterior por trial (auditada)" if v == "preceding_pause" else "Intervalo fixo do JSON")
        if not cfg["windows"] or not cfg["offsets"] or not cfg["bands"] or not cfg["svm_c"]:
            raise ValueError("Selecione ao menos uma janela, offset, banco de bandas e valor de C.")

        chosen = selection(control, subjects, sessions, blocks, runs)
        st.write(f"{len(chosen)} registros selecionados.")
        if not chosen.empty:
            st.dataframe(chosen[["run_id", "subject", "session", "block_type"]], hide_index=True)
        if st.button("Verificar dados antes da análise", disabled=chosen.empty):
            from analyze import check_dataset
            output = Path(out_text).expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            with st.spinner("Verificando arquivos, relógio, marcadores, épocas e viabilidade dos protocolos..."):
                diagnostics = check_dataset(root, control, output, cfg, subjects, sessions, blocks, runs, protocols)
            st.write(f"Diagnóstico salvo em {output}")
            st.dataframe(pd.DataFrame([dict(run_id=r["run_id"], **c) for r in diagnostics for c in r["checks"]]), hide_index=True)
            st.download_button("Baixar diagnóstico JSON", (output / "diagnostics.json").read_bytes(), "diagnostics.json", "application/json")
        if st.button("Gerar relatórios", type="primary", disabled=chosen.empty or (mode != "ephys" and not protocols)):
            from analyze import batch
            output = Path(out_text).expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            with st.spinner("Processando. Busca aninhada pode demorar; resultados serão gravados por registro."):
                results = batch(root, control, output, cfg, subjects, sessions, blocks, runs, mode, detailed, protocols, compare)
            st.session_state["last_output"] = str(output)
            st.success(f"Execução concluída: {output}")
            st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k != "classification"} for r in results]), hide_index=True)
    except Exception as exc:
        st.error(str(exc))
with tab_results:
    st.header("4. Consultar resultados")
    output_root = Path(out_text).expanduser()
    choices     = sorted({p.parent for pattern in ["*/batch_results.json", "*/diagnostics.json"] for p in output_root.glob(pattern)}, reverse=True) if output_root.exists() else []
    if not choices:
        st.info("Gere um relatório para consultar os resultados aqui.")
    else:
        output  = st.selectbox("Execução", choices, format_func=lambda p: p.name)
        if (output / "diagnostics.json").exists():
            diagnostic_rows = json.loads((output / "diagnostics.json").read_text())
            with st.container():
                st.dataframe(pd.DataFrame([dict(run_id=r["run_id"], **c) for r in diagnostic_rows for c in r["checks"]]), hide_index=True)
                st.download_button("Baixar diagnóstico completo", (output / "diagnostics.json").read_bytes(), "diagnostics.json", "application/json", key="saved_diagnostics")
        results = json.loads((output / "batch_results.json").read_text()) if (output / "batch_results.json").exists() else []
        visible = [r for r in results if (not subjects or r["subject"] in subjects)
                   and (not sessions or r["session"] in sessions) and (not blocks or r["block_type"] in blocks)
                   and (not runs or r["run_id"] in runs)]
        if (output / "index.html").exists():
            bundle = BytesIO()
            with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
                for artifact in sorted(output.rglob("*")):
                    if artifact.is_file() and artifact.suffix in {".html", ".csv", ".json", ".png", ".pdf", ".txt"}:
                        archive.write(artifact, artifact.relative_to(output))
            st.download_button("Baixar batch completo (ZIP com relatórios e auditoria)", bundle.getvalue(), "relatorios.zip", "application/zip")
            st.download_button("Baixar relatório geral do batch", (output / "index.html").read_bytes(), "index.html", "text/html")
        for result in visible:
            with st.container():
                st.subheader(result["run_id"])
                if result["status"] != "ok":
                    st.error(result["status"])
                    continue
                folder = output / result["run_id"]
                st.caption("Baixe e abra o HTML no navegador para zoom, inspeção do raw e traçados contínuos. As imagens abaixo são resumos estáticos.")
                st.image(str(folder / "dashboard.png"))
                st.download_button("Baixar relatório HTML completo", (folder / "report.html").read_bytes(),
                                   file_name=f"{result['run_id']}.html", mime="text/html", key=result["run_id"])
                st.dataframe(pd.DataFrame(result.get("classification", [])), hide_index=True)
