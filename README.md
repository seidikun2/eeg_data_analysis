# EEG/BCI · análise local

Conversão do `batch_analysis.ipynb` para execução local, sem Colab ou Google Drive.
Quatro arquivos de código: `eeg.py` (dados/controle), `analyze.py` (eletrofisiologia e relatórios),
`classify.py` (modelos/validação) e `app.py` (interface). Os dados permanecem na pasta escolhida.

## Instalação e interface

Python **3.12** recomendado. No terminal, dentro deste repositório:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Na interface:

1. Informe a pasta local dos dados e clique em **Atualizar índice**.
2. Em **Controle dos dados**, marque `include`, revise o tipo de bloco e salve.
3. Selecione sujeitos, sessões, blocos ou registros na barra lateral; vazio significa todos.
4. Em **Produzir relatório**, escolha eletrofisiologia, classificação ou ambas. Ative a inspeção
   de cada época para relatórios minuciosos. Escolha os protocolos e gere os relatórios.
5. Consulte as imagens e baixe o HTML em **Consultar resultados**. Também pode abrir
   `analysis_outputs/<execução>/index.html` diretamente no navegador.

A interface executa a análise localmente e não faz upload de gravações. Uma análise extensa
pode demorar; comece com um registro e `left_right`. A busca padrão avalia até 12 candidatos
por fold, não toda a grade. Cada execução cria sua própria pasta, evitando misturar resultados.
Os gráficos são PNG a 300 dpi e PDF vetorial; o HTML MNE contém as figuras e tabelas.

## Formato de entrada e controle manual

Formato preservado do notebook:

```text
dados/AB01/S1/IM_treino/..._signal_20260101_120000.csv
dados/AB01/S1/IM_treino/..._markers_20260101_120000.csv
```

O sinal requer `lsl_time_s`, `ch1` até `ch15`; os marcadores requerem `lsl_time_s`, `label`.
O par deve diferir apenas por `_signal_` / `_markers_`. O mapa de canais é o mesmo do notebook:
FC3, FCz, C5, C3, FC4, C1, CP3, Cz, C2, C4, CP1, CPz, CP4, CP2, C6.
Usa-se a montagem MNE `colin27_1020`, ou seu nome legado equivalente `standard_1020`
quando o primeiro não existe na versão instalada. São posições padronizadas, não digitalização
individual. O QC registra a montagem escolhida e o diagnóstico inclui as versões das bibliotecas,
mesmo quando o pré-processamento falha.
Os topomapas usam interpolação linear local sobre uma cobertura sensorimotora limitada,
evitando extrapolar para todo o escalpo ou introduzir extremos por interpolação cúbica.

**Confirme a unidade dos CSVs**: `scale_to_volts=1e-6` presume µV, como no notebook.
Se os valores já estiverem em volts, use `1.0`. Os timestamps de sinal e marcadores
precisam compartilhar o relógio LSL e estar em segundos.

`dataset_index.csv` pode ser editado pela interface ou por uma planilha:

| Campo | Uso |
| --- | --- |
| `include` | `True` inclui, `False` exclui; novos registros começam excluídos |
| `block_type` | `IM_treino`, `EM_treino`, `online_com_feedback`, `online_sem_feedback` |
| `group` | Rótulo de grupo preservado nos metadados; não gera inferência estatística automática |
| `bad_channels` | Canais a excluir, separados por vírgula, por exemplo `C3,CPz` |
| `exclude_trials` | IDs originais a excluir, **começando em zero**, por exemplo `0,3,8` |
| `reason` | Justificativa manual, registrada no relatório |
| `train_run_id` | Opcional: registro IM anterior usado como calibração proxy para online |

Online sem identificação de feedback é tratado como `online_com_feedback`.
O valor legado `online_definir` também é convertido ao ler ou atualizar a tabela, preservando `include`.
Sujeito/sessão são inferidos como no notebook (`AB01`, `S1`); revise `UNKNOWN` na tabela
se a organização de pastas for diferente. Caminhos novos são relativos à pasta dos dados,
permitindo movê-la entre máquinas. Atualizar o índice preserva decisões manuais, mantém
registros desaparecidos com `available=False` e não remove silenciosamente exclusões.
Pares incompletos são indexados e produzem erro explícito se incluídos.

## Execução por terminal

```bash
python analyze.py index --root "/caminho/dados"
# Editar dataset_index.csv antes de continuar.
python analyze.py batch --root "/caminho/dados" --subjects AB01 AB02 --mode ephys
python analyze.py batch --root "/caminho/dados" --subjects AB01 --blocks IM_treino --mode classification --protocols left_right
python analyze.py report --root "/caminho/dados" --subjects AB01 --sessions S1 --protocols left_right rest_move
```

`batch` produz resumo e figuras por registro; `report` acrescenta cada época antes/depois.
Sem filtros, entram todos os registros incluídos e com tipo de bloco definido.
Também existem `--runs`, `--control`, `--output` e `--config`. As duas análises podem ser
executadas independentemente com `--mode ephys` ou `--mode classification`.
O comando termina com código 1 se um registro ou protocolo solicitado não puder ser analisado;
os resultados bem-sucedidos são preservados e o motivo consta no HTML/JSON.

## Check antes do batch

Na interface, clique em **Verificar dados antes da análise**, depois de salvar a tabela.
Pelo terminal:

```bash
python analyze.py check --root "/caminho/dados" --subjects AB01
```

O check não treina modelos. Ele verifica os dois CSVs independentemente, leitura e colunas,
relógio, marcadores nas bordas, pré-processamento, sequência/duração de fases, rejeição de
épocas, baseline em PAUSA, classes e disponibilidade de folds por trial. Candidatos com
janelas/offsets que não suportam CV são descartados antes de calcular a interseção de trials;
um candidato inviável não deve inutilizar os candidatos mais curtos.

Saídas: `diagnostics.html`, `diagnostics.csv`, `diagnostics.json` e um JSON por registro.
Cada item informa etapa, `ok`/`warning`/`error`, mensagem e ação recomendada. O JSON contém
configuração, QC, durações das fases e rejeições. O diagnóstico é salvo por registro, mesmo
que outros registros falhem; arquivos ausentes também aparecem. Apenas registros incluídos
pelos filtros e controle são verificados. O check é executado automaticamente antes do batch.
Falhas posteriores da análise geram também `error_traceback.txt` no diretório do registro.

Marcadores ligeiramente anteriores/posteriores ao EEG não bloqueiam o bloco inteiro:
seus tempos originais são preservados e a distância às bordas é registrada. Trials precisam
ter ATTEMPT dentro do sinal; épocas sem EEG suficiente são rejeitadas pelo MNE. Na classificação,
o segmento é limitado às amostras disponíveis. Se nenhum marcador sobrepõe o EEG, a leitura
continua bloqueada para investigar sincronização.

Quando a baseline não cai em PAUSA, o check procura um intervalo comum nas PAUSAs anteriores
com margem de 250 ms e duração mínima de 500 ms, e registra `suggested_erd_config` quando possível.
A sugestão inclui ampliar o contexto pré-ATTEMPT; **não é aplicada automaticamente**. Revise-a
em uma cópia do JSON e execute novamente o check. O primeiro trial pode continuar sem baseline.
Um check `ok` não certifica ausência de artefatos nem substitui inspeção ou ajuste numérico do modelo.

## Eletrofisiologia

- Sanity checks: finitude, monotonicidade, amostragem/jitter/gaps/drift, limites dos marcadores,
  reconstrução das fases, canais ruins, amplitude, desvio-padrão e fração de passos constantes.
  Por padrão, irregularidades pequenas dos timestamps são tratadas ajustando um relógio
  uniforme à ordem original das linhas (sem excluir, ordenar ou interpolar amostras EEG).
  Isso pressupõe que as linhas preservam a ordem de aquisição e que não houve perda de amostras.
  Timestamps duplicados podem ocorrer em gravação por chunks; apenas timestamps não permitem
  provar se houve perda de dados. O sanity check registra repetições, reversões, taxa estimada
  e o maior ajuste. Não se imputam NaNs.
  `timestamp_policy="strict"` mantém a recusa anterior. No modo padrão `sample_clock`,
  correções maiores que `max_timestamp_correction_s=0.1` s são recusadas para investigar
  gaps/reinícios de relógio. Esse limite é uma tolerância configurável, não uma garantia de
  precisão dos eventos. A reconstrução pode alterar o alinhamento de eventos dentro desse
  limite; confira o diagnóstico antes de interpretar ERP ou timing fino.
- Pré-processamento MNE: notch opcional, filtro FIR offline 0,5–45 Hz, referência média configurável.
  A referência usa canais bons; os canais marcados ruins são excluídos, sem interpolação.
  A escolha de 0,5 Hz preserva mais informação para inspeção de ERP do que 5 Hz no notebook,
  embora ainda possa alterar componentes lentos. Não há ICA automático ou alegação de remover todo artefato.
- Épocas alinhadas ao **ATTEMPT**, de −3,5 a +5,25 s, rejeição por pico a pico/flatness,
  sinal bruto antes/depois, imagens de épocas por canal e ERP de todos os canais no mesmo eixo.
  A cor identifica classe; canais não recebem cores arbitrárias.
- Welch por classe no intervalo ativo; espectrograma contínuo; Morlet por trial para ERD/ERS,
  com baseline individual antes da média. Os gráficos de banda mostram a média das variações
  relativas por frequência/canal, não a média do sinal antes de calcular potência.
- ERD negativo = redução de potência; ERS positivo = aumento. Topografias mu [8,13) e beta
  [13,30) Hz usam o mesmo limite simétrico entre classes/bandas. A grade padrão é de 2 Hz;
  os limites incluem as frequências disponíveis nesse intervalo, não integração espectral contínua.
- PCA 2D/3D sobre covariâncias OAS → espaço tangente → padronização, uma observação por trial.
  É exploratório, ajustado no conjunto visualizado, e não fornece desempenho de generalização.

**Baseline:** `baseline` é relativa ao ATTEMPT e precisa caber inteiramente em uma PAUSA
anterior conhecida para entrar no ERD/ERS. O primeiro trial geralmente não atende a isso.
O relatório salva `baseline_audit.csv`; se nenhum trial atender, ERD/TFR/topografias ficam
indisponíveis, com explicação. Não se substitui PAUSA por ATTENTION silenciosamente.
A baseline do ERP continua sendo o intervalo configurado, mesmo quando não é uma PAUSA,
pois o ERP é apenas um check descritivo. `active` é o intervalo usado em PSD, PCA e topografias.

## Classificação e escolha de parâmetros

Pipeline principal: **Covariância OAS → TangentSpace Riemann → StandardScaler → SVM RBF**.
Banco de bandas concatena features tangentes de cada banda. Para experimentar CSP+SVM,
use `"pipelines": ["tgsp", "csp"]` no JSON; o CSP também é ajustado só nos folds de treino.
`config.json` define janelas, offsets, passo, bandas, C, gamma, folds e orçamento de busca.
Um JSON parcial passado com `--config` substitui apenas os campos fornecidos. Exemplo:

```json
{
  "windows": [1.0, 1.5, 2.0],
  "offsets": [0.0, 0.5, 1.0],
  "bands": [[[5, 35]], [[8, 13], [13, 30]]],
  "pipelines": ["tgsp", "csp"],
  "svm_c": [0.1, 1.0, 10.0],
  "svm_gamma": ["scale", 0.01],
  "search_budget": 24
}
```

O offset é aplicado ao início de cada fase usada; as janelas nunca ultrapassam a fase.
A duração de cada fase é limitada a `attempt_duration` para uma PAUSA longa não dominar
o treino. O filtro do classificador é Butterworth causal de ordem 4, por trial isolado,
com notch causal opcional e referência configurada. Usa-se o sinal bruto, sem reutilizar
o FIR contínuo offline. Isso evita mistura de amostras de trials de teste no filtro do treino.
Não há decimação por slicing; mantém-se a taxa original, evitando aliasing.
Filtros, referência e calibração diferem potencialmente do sistema online de coleta:
estas análises não são uma reprodução garantida do classificador implantado.

A unidade de separação é o **trial inteiro**. Todas as janelas/fases de um trial ficam
do mesmo lado de cada divisão, com estratificação pela condição original do trial.
A busca é aninhada: parâmetros escolhidos no CV interno, desempenho no teste externo.
O scaler, espaço tangente, CSP e calibração sigmoid são ajustados dentro dos folds.
As probabilidades usam `CalibratedClassifierCV` com divisões por trial; não se usa a
calibração interna padrão do SVM sobre janelas sobrepostas. Poucos trials causam um
impedimento explícito, nunca uma troca silenciosa para CV por janelas.

A busca interna usa BA de votos por trial/fase. A BA final e a matriz de confusão usam
probabilidades médias por trial/fase; BA por janela é salva separadamente. Assim uma
fase longa não pesa como vários trials independentes. A mesma interseção de unidades
trial/classe é usada para comparar candidatos viáveis. O JSON registra os candidatos
inválidos, scores internos, parâmetros escolhidos e IDs de treino/teste de cada fold.
Os scores internos **não** são desempenho independente de cada candidato.

Protocolos preservados:

| Protocolo | Classes/fases |
| --- | --- |
| `left_right` | LEFT × RIGHT durante ATTEMPT |
| `rest_move` | PAUSA/REST × ATTEMPT de LEFT/RIGHT/BOTH; inclui REST após REST_STIM |
| `attention_move` | ATTENTION × ATTEMPT de LEFT/RIGHT/BOTH |
| `reststim_move` | REST_STIM × ATTEMPT de LEFT/RIGHT/BOTH |
| `rest_left_right` | LEFT × RIGHT × REST; usa REST_STIM se existir, senão PAUSA |

A fonte da terceira classe fica registrada em `rest_source`: REST_STIM e PAUSA não são
fisiologicamente equivalentes. Protocolos com classes ausentes são declarados indisponíveis.

Nos blocos online, usa-se apenas IM_treino incluído, do mesmo sujeito, **estritamente anterior**.
Sem indicação manual, busca-se o mais recente da mesma sessão. Outro registro anterior
do mesmo sujeito pode ser indicado por `train_run_id`. A busca de parâmetros e calibração
ocorre apenas nesse treino; o bloco online fica reservado ao teste. A origem é sempre
declarada **proxy**, pois o notebook não contém a calibração real usada na coleta.

Curvas mostram somente predições de teste (ou teste online proxy), posicionadas no **fim
da janela causal**. Trials antigos têm alpha menor e recentes, maior; a média recebe linha
mais forte. Não há suavização centrada, extrapolação, normalização da duração do trial
ou interpolação através de fases sem amostras. Quando folds escolhem janelas diferentes,
a média usa os tempos de fim efetivamente coincidentes; suporte pode variar ao longo da curva.
Uma tendência visual não constitui evidência causal de aprendizado.

## Saídas e reprodutibilidade

Cada execução contém `index.html` (agregador), `selected_control.csv`, `summary.csv` e
`batch_results.json`. Cada registro tem um dashboard em uma única imagem, imagens/PDFs,
`report.html` autocontido, tabelas de trials/rejeição/baseline, parâmetros, métricas e
predições por janela. A configuração, versões das bibliotecas, linha de controle e
hashes SHA-256 dos dois CSVs ficam em `parameters_qc.json`.

O dashboard resume raw, ERP, espectro, ERD/ERS, topografias mu/beta de LEFT/RIGHT quando
disponíveis, PCA e classificação; as demais classes/protocolos ficam nas figuras completas.
Os relatórios detalhados mostram também o sinal processado de épocas rejeitadas.
Épocas sem contexto suficiente nas bordas são registradas como rejeitadas, sem padding.
Não se juntam sessões com canais/taxas diferentes em um único modelo, nem se atribui
significância a centenas de janelas dependentes. Comparações entre sujeitos exigem
definir desenho experimental e unidade estatística antes de um teste inferencial.

Testes de regressão, usando apenas CSVs sintéticos:

```bash
python -m pip install pytest
python -m pytest -q
```

Os testes verificam reconstrução de fases, preservação de exclusões, reconstrução de timestamps repetidos, recusa de reinícios de relógio
e modo estrito e isolamento de trials no CV, além da interface e geração de relatórios.
**Ainda é necessário validar uma gravação real**: unidade física, mapa dos eletrodos,
tempos efetivos do protocolo, baseline, referência e limiares de rejeição.
