# EEG/BCI · análise local

Análise dos CSVs `signal/markers`, com controle manual, relatórios individuais com zoom
e comparação batch entre sujeitos e pipelines. Dados permanecem na pasta escolhida.

## Iniciar

Python 3.12 recomendado, dentro da pasta do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

No Windows, ative com `.venv\Scripts\Activate.ps1`.
Após atualizar o código, instale as dependências e reinicie o Streamlit. Relatórios antigos
não são modificados: gere uma nova execução para usar o novo formato.

## Fluxo na interface

1. Informe a pasta dos dados e clique em **Atualizar índice**.
2. Em **Controle dos dados**, marque `include`, revise `block_type` e salve.
3. Filtre sujeitos, sessões, blocos e registros na barra lateral; vazio significa todos.
4. Execute **Verificar dados antes da análise**. O check informa erro/aviso e ação por registro.
5. Em **Gerar relatório**, escolha eletrofisiologia, classificação ou ambas.
6. Para comparação, marque **Batch comparativo: avaliar TGSP e CSP separadamente**.
   Em **Ajustar experimento e visualização**, selecione janelas, offsets, bandas/bancos, C do SVM e orçamento de busca.
7. Gere os relatórios. Em **4. Consultar resultados**, baixe o HTML individual ou o comparativo.

Abra o HTML no navegador: os gráficos interativos têm zoom por seleção, roda, pan e
restauração por duplo clique. A legenda permite ocultar traços. Figuras estáticas abrem
em um visualizador com zoom e arraste. Tudo funciona offline, incluindo os gráficos Plotly.
As seções ficam visíveis em ordem previsível. Detalhes extensos permanecem nos CSVs/JSONs de auditoria.

## Batch comparativo

Sem selecionar um sujeito, o batch processa todos os registros incluídos. Restrinja um tipo
de bloco para facilitar comparações equivalentes. Comece por `left_right` para conferir os dados.

```bash
python analyze.py index --root "/caminho/dados"
# Editar include e block_type em dataset_index.csv.
python analyze.py check --root "/caminho/dados"
python analyze.py batch --root "/caminho/dados" --blocks IM_treino --mode both --protocols left_right --compare-pipelines
python analyze.py report --root "/caminho/dados" --subjects AB01 --sessions S1 --protocols left_right
```

`batch` gera resumo e figuras por registro; `report` acrescenta cada época antes/depois.
Os filtros `--subjects`, `--sessions`, `--blocks` e `--runs` são opcionais. Também existem
`--config`, `--control`, `--output` e `--mode ephys|classification|both`.

`index.html` é o relatório geral completo: BA em pontos por sujeito/sessão/registro/pipeline, evolução do ERD/ERS por tentativa e links dos relatórios individuais. `comparison.html` redireciona ao geral por compatibilidade.
`comparison_runs.csv` conserva cada registro; `comparison_subjects.csv` agrega as repetições
dentro de cada sujeito; `pipeline_paired_differences.csv` contém diferenças pareadas de BA.
`erd_subjects.csv` agrega primeiro os trials por registro e depois os registros por sujeito.
Janelas e folds não contam como sujeitos independentes. As comparações são descritivas,
sem teste de significância automático; não estimam generalização para sujeitos inéditos.
Blocos, protocolos, CV e avaliação online proxy ficam separados.

Com a comparação ativada, cada pipeline tem sua própria busca interna e avaliação externa.
Sem ela, `pipelines` no JSON controla uma pipeline ou uma seleção aninhada entre várias.
O orçamento limita candidatos por busca; `comparison_pipelines` define as famílias a comparar.

## O que aparece no relatório individual

- **Resumo:** trials, épocas aceitas/rejeitadas, exclusões manuais e canais excluídos.
- **Sanity checks:** variação antes/depois, offsets DC em µV, amplitudes pico a pico,
  contribuição de cada canal para rejeições, potência estreita de 50/60 Hz e alertas de
  variação relativa (<0,2× ou >5× a mediana dos canais). Alertas não excluem automaticamente.
- **Raw:** snapshot de 12 s no meio do bloco; antes em cinza e depois em azul, sobrepostos
  por canal. A mediana de cada traço é removida **apenas para exibição**. O eixo continua em µV.
  A faixa inicial de exibição usa P99,5; extremos podem ficar fora do enquadramento e podem
  ser examinados ajustando o eixo interativo. Os valores completos ficam nas tabelas de QC.
- **ERP:** uma grade por canal e classe, trials discretos e média preta destacada. Escalas por canal, identificadas em µV; não se colorizam canais.
- **ERD/ERS:** mu/beta por C3/Cz/C4 e classe, trials individuais e média preta, tempo–frequência alinhado ao ATTEMPT por canal e
  topografias com escala comum. Negativo = redução; positivo = aumento de potência.
- **Classificação:** matriz de confusão, scores externos, timeline contínua de teste,
  curvas por época com alpha crescente por cronologia e média destacada.
- **Detalhes:** PCA 2D/3D com janelas reais dependentes e IDs de trial, épocas conjuntas antes/depois, motivos de rejeição, configuração,
  versões e hashes dos arquivos. Épocas rejeitadas também podem ser inspecionadas.

Paleta de classes: Okabe–Ito com azul/laranja/verde/roxo/cinza; mapas sequenciais em cividis
ou cinza, mapas divergentes em azul/vermelho. Cores de classes são consistentes nas figuras.

## Baseline e pré-processamento

Confirme `scale_to_volts`: o padrão `1e-6` presume CSVs em **µV**, como no notebook.
Se já forem volts, use `1.0`. Canal e unidade não são inferidos pela amplitude.

MNE: filtro FIR offline 0,5–45 Hz, notch opcional, referência média configurável,
rejeição por pico a pico/flatness. Não há ICA automático ou garantia de remover todos os artefatos.
Canais ruins são excluídos, sem interpolação. Montagem padronizada `colin27_1020`, com
nome legado equivalente `standard_1020` para MNE antigo; não é digitalização individual.

Por padrão `erd_baseline_mode="preceding_pause"`: para cada trial procura-se até 1,5 s
na PAUSA anterior, com margens de 250 ms e mínimo de 500 ms. O contexto necessário é lido
do EEG contínuo, verificado por amplitude/flatness, e a normalização é feita por trial antes
da média. `baseline_audit.csv` registra intervalos e exclusões. O primeiro trial geralmente
não tem referência válida. Se nenhum trial servir, o relatório explica a ausência de ERD.
`erd_baseline_mode="fixed"` usa o intervalo `baseline` relativo ao ATTEMPT, exigindo que esteja
em PAUSA. O ERP sempre usa `baseline`; é um check descritivo, sem alegação de resposta evocada.
`active` controla o intervalo de PSD/PCA/topografias. Topomapas usam interpolação linear local
sobre cobertura sensorimotora limitada. Mu é [8,13) Hz e beta [13,30) Hz na grade TFR configurada.

## Validação dos classificadores

TGSP: OAS → espaço tangente Riemann → StandardScaler → SVM RBF.
Alternativa: CSP com regularização OAS → StandardScaler → SVM RBF.
Banco de bandas concatena features por banda. Janelas, offsets, bandas, C e gamma são configuráveis.
Um JSON parcial passado com `--config` altera só os campos fornecidos:

```json
{
  "windows": [1.0, 1.5, 2.0],
  "offsets": [0.0, 0.5, 1.0],
  "bands": [[[5, 35]], [[8, 13], [13, 30]]],
  "svm_c": [0.1, 1.0, 10.0],
  "search_budget": 24,
  "curve_smoothing_s": 0.6
}
```

O trial inteiro é separado entre treino/teste, com estratificação pela condição original.
A busca é aninhada; features, scaler e calibração sigmoid são ajustados no treino.
A BA/matriz final usa probabilidades médias por trial/fase; BA por janela é registrada separadamente.
A busca interna usa votos por trial/fase. Candidatos sem suporte temporal/classes/CV são descartados
antes de calcular a interseção de unidades entre candidatos. Poucos trials geram impedimento explícito.

O classificador usa Butterworth causal ordem 4 por trial, notch opcional e referência configurada,
sem reutilizar o FIR offline entre trials. A timeline usa o modelo do fold em que o trial ficou
no teste. Janelas atravessam as fases dentro do trial para mostrar o contexto STIMULUS e transições,
mas nenhuma janela atravessa a fronteira entre trials. Trials de condições fora do protocolo
(ex. REST_STIM em LEFT×RIGHT), que não entram no treino, recebem predição de um fold designado
somente como contexto (`context_only`); não entram nas métricas. Não há predição para trechos sem suporte
nem para trials rejeitados. Isso pode deixar lacunas reais entre trials.

As curvas de épocas são extraídas dessa timeline; a média usa interpolação linear apenas dentro
do suporte de cada trial. Não se extrapola para preencher bordas; o hover informa quantos trials
contribuem em cada tempo. `curve_smoothing_s` aplica média móvel **causal por trial/fold**, apenas
à visualização. Probabilidades e métricas exportadas permanecem sem suavização. A janela é
posicionada em seu fim; suavização e duração da janela introduzem atraso visual.

Online: somente IM_treino incluído, do mesmo sujeito e estritamente anterior; por padrão o mais
recente da mesma sessão. `train_run_id` pode indicar outro anterior. Sempre é identificado como
**proxy**, não como reprodução garantida do modelo usado na coleta. Tendências visuais não provam aprendizado.

Protocolos: `left_right`, `rest_move`, `attention_move`, `reststim_move`, `rest_left_right`.
REST é PAUSA; REST_STIM é condição de ATTEMPT. No multiclasses, a fonte de REST é registrada.
Online sem identificação e o valor legado `online_definir` entram como `online_com_feedback`.

## Dados, controle e diagnóstico

Organização típica: `dados/AB01/S1/IM_treino/..._signal_<timestamp>.csv` e par `_markers_`.
Sinal: `lsl_time_s`, `ch1`…`ch15`. Marcadores: `lsl_time_s`, `label`. Mapa preservado do notebook:
FC3, FCz, C5, C3, FC4, C1, CP3, Cz, C2, C4, CP1, CPz, CP4, CP2, C6.

No `dataset_index.csv`: `include`, `block_type`, `group`, `bad_channels` (ex. `C3,CPz`),
`exclude_trials` (IDs a partir de zero, ex. `0,3,8`), `reason` e `train_run_id` são editáveis.
Atualizar índice preserva decisões; novos registros começam excluídos. Caminhos são relativos
à raiz. O grupo é metadado; não gera inferência estatística automática.

O check salva `diagnostics.html/csv/json`, mesmo com arquivos ausentes, e o batch conserva
`error_traceback.txt` se falhar depois. Timestamps pequenos repetidos/reversos podem ser
reconstruídos sob hipótese de ordem de aquisição e taxa uniforme, preservando amostras;
limite padrão de ajuste: 100 ms. O ajuste é auditado e não garante sincronização. Use
`timestamp_policy="strict"` para recusar irregularidades. Marcadores nas bordas são registrados,
sem deslocamento; épocas incompletas são descartadas. Relógios sem sobreposição continuam bloqueados.

Cada execução tem pasta própria, `index.html`, `comparison.html`, CSVs e JSONs. Por registro:
`report.html` autocontido, PNG 300 dpi/PDF, `parameters_qc.json`, `channel_quality.csv`,
`trials.csv`, `epoch_audit.csv`, `baseline_audit.csv`, `erd_metrics.csv`, métricas e predições.
`*_timeline.csv` guarda probabilidades não suavizadas do contexto de teste.

Testes: `python -m pip install pytest && python -m pytest -q`. São testes com dados sintéticos;
valide unidades, mapa dos canais, protocolo e limiares em suas gravações reais.

## Apresentação consolidada

Fluxo visível: selecionar dados/filtros → verificar qualidade → gerar individual ou geral → consultar.
Os parâmetros avançados ficam em uma única área secundária (janelas, offsets, bancos, C e gamma do SVM).
O download ZIP preserva os links entre o geral, os individuais e os arquivos de auditoria. Abra `index.html` após extrair.
O relatório individual reúne processamento, qualidade, snapshot central, ERP, ERD/ERS, topografias e classificação.
A timeline ocupa a largura inteira, com ATTEMPT sombreado e classe/ID indicados. Zoom por seleção/roda e
figuras ampliáveis funcionam offline. Lacunas entre trials continuam sem conexão; suavização é apenas visual.
Folds aparecem como dispersão dependente, sem linhas sequenciais. PCA usa janelas reais de até 1 s
dentro do intervalo ativo: linhas e IDs mantêm a identidade do trial; não aumenta a amostra independente.

Validação: `python -m pytest -q` inclui batch real com dois sujeitos sintéticos e TGSP/CSP,
auditoria de separação treino/teste por trial e intervalos da timeline. Não substitui validação nas gravações reais.
