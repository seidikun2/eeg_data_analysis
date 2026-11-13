# EEG/BCI · análise local por projeto, sujeito, sessão e registro

Python 3.12 recomendado:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m streamlit run app.py
```

No ambiente cloud preparado, use `source /workspace/venvs/eeg_data_analysis/bin/activate`.
Se a home não for gravável: `export MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache`.
Dados e relatórios permanecem locais. Não versionar gravações ou identificações pessoais.

## Fluxo

1. **Controle dos dados:** escolher raiz, atualizar índice, revisar `include`, tipo de bloco, canais/trials excluídos e fonte de treino.
2. **Verificar qualidade:** conferir arquivos, timestamps, marcadores, épocas, baseline e viabilidade da CV.
3. **Produzir relatório:** selecionar protocolos, TGSP/CSP e classificadores. Uma única área avançada reúne janelas, offsets, bancos de bandas, C/gamma, orçamento, baseline e decimação.
4. **Consultar resultados:** baixar check individual, sessão ou ZIP completo. Extrair o ZIP e abrir `index.html` para preservar os links.

A estimativa inicial de duração é **heurística**, baseada no número de registros, candidatos, protocolos e folds. Após cada registro, o tempo restante é recalculado pelo custo médio observado. Duração, canais e carga da máquina alteram a precisão; o primeiro registro pode demorar mais. Modelos não são ajustados no check de qualidade.

## Dados: Brasil e Italia

```text
dados/
  Projeto Brasil/AB01/S1/IM_treino/..._signal_20260101_120000.csv
  Projeto Brasil/AB01/S1/IM_treino/..._signal_20260101_123000.csv
  Projeto Brasil/AB01/S1/online_com_feedback/..._signal_20260101_130000.csv
  Projeto Italia/AB01/S1/EM_treino/..._signal_20260101_110000.csv
```

Cada sinal requer seu par `_markers_`. Repetições têm timestamps diferentes e permanecem separadas; pastas como `IM_treino_2` também são reconhecidas. Não se unem sujeitos, sessões ou repetições para produzir uma média de performance. O projeto participa da identidade; a seleção automática de calibração nunca atravessa projetos.

Sinal: `lsl_time_s`, `ch1`…`chN`. Marcadores: `lsl_time_s`, `label` (ATTENTION, condição `_MI_STIM`, ATTEMPT, PAUSA/PAUSE/REST). A raiz antiga sem pasta de projeto continua sendo Brasil.

Brasil mantém o mapa de 15 canais: FC3, FCz, C5, C3, FC4, C1, CP3, Cz, C2, C4, CP1, CPz, CP4, CP2, C6.

Italia usa, na ordem, até 53 canais:

```text
Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz C4 T8 M2
CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 AF7 AF3 AF4 AF8 F5 F1 F2 F6 FC3 FCz
FC4 C5 C1 C2 C6 CP3 CP4 P5 P1 P2 P6 FT7 FT8 TP7 TP8
```

Um canal além do mapa só é descartado como possível TRIGGER se estiver **inteiramente zerado**; a decisão fica no QC. Um extra não zerado ou sem identificação bloqueia a leitura, em vez de inventar um nome. Para outro arranjo, forneça `channel_names` no JSON. Menos canais pressupõe os primeiros N dessa lista; se houve remoção/reordenação física, forneça a lista exata. A montagem usa `colin27_1020`, com fallback `standard_1020` em versões anteriores do MNE.

O índice preserva decisões manuais e mantém registros ausentes como indisponíveis. `include`, `block_type`, `group`, `bad_channels`, `exclude_trials`, `reason` e `train_run_id` são editáveis. Novos registros começam excluídos. IDs de trials começam em zero. Não se excluem canais automaticamente nem se aumentam limiares para fazer um teste passar.

## Relatórios

- `index.html`: geral com épocas **totais, elegíveis, aceitas, rejeitadas e excluídas**, qualidade, BACC, CV de parametrizações fixas, com figuras por janela, offset, bandas e modelo. Cada sujeito e cada repetição têm painéis próprios. Protocolos como `left_right` e `reststim_move` aparecem lado a lado quando selecionados.
- `session_<projeto>_<sujeito>_<sessão>.html`: sessão completa; qualidade e performance lado a lado na tabela, seguida dos registros completos, mantendo as repetições. EM_treino é screening de movimento; IM_treino treina; blocos online testam.
- `<run_id>/check.html`: check individual com informações básicas, processamento, qualidade e snapshot. `<run_id>/report.html` preserva a inspeção completa incorporada à sessão.
- `comparison.html`: entrada de compatibilidade que redireciona ao geral.

O snapshot mostra até **20 s do meio do bloco**, com canais em offset e escala comum em µV: antes cinza, depois vermelho. Apenas a mediana visual é removida. O desenho usa P99,5 para não ser dominado por extremos; valores fora do enquadramento não são alterados nos dados ou no QC. ERP e ERD têm classes separadas e escala comum entre canais. ERD/ERS mostra a linha zero em cada offset, sem mapa de trials de um único canal. A figura conjunta de épocas antes/depois não aparece na apresentação; `epoch_audit.csv` mantém a auditoria de rejeição; com inspeção detalhada, `epoch_audit.png` guarda a figura fora do HTML. O batch não apresenta tendências ERD/ERS; `erd_subjects.csv` mantém os valores por trial.

As tabelas mantêm os valores numéricos de BACC; a cor é apenas um auxílio. Figuras estáticas podem ser ampliadas. Timeline e PCA são interativos, offline, com zoom e legenda para isolar classes. Parâmetros de seleção, BACC e timeline ficam na mesma seção da combinação de features/classificador. A timeline usa a largura inteira dessa seção.

## Processamento e baseline

Confirme `scale_to_volts`: o padrão `1e-6` presume CSV em **µV**; para volts, use `1.0`. Não se inferem unidades pela amplitude.

Eletrofisiologia: MNE FIR offline 0,5–45 Hz, notch opcional, referência média configurável, rejeição por pico a pico/flatness. Sem ICA automático ou interpolação de canais ruins. Os sanity checks incluem ruído de 50/60 Hz **antes da decimação**, variações relativas por canal, amplitudes e rejeições.

`analysis_decimation="auto"` usa um em cada dois pontos a partir de 250 Hz, após antialias Butterworth **causal** de ordem 8. Isso evita aliasing de frequências acima do novo Nyquist. `1` desativa; `2` força, desde que `h_freq` caiba no novo Nyquist. Taxas de origem/análise e fator ficam no relatório/QC; marcadores continuam no relógio original. A análise passa a usar o sinal antialiasado, inclusive nos snapshots; a triagem de ruído de rede usa o sinal de aquisição.

Por padrão, ERP e ERD referenciam **ATTENTION marcado**, do início de ATTENTION +100 ms até STIMULUS −100 ms, exigindo pelo menos 500 ms dentro do EEG. ERP é descritivo; ERD é normalizado por trial antes da média. ERD usa também critérios de amplitude/flatness no contexto da baseline. ATTENTION não é repouso fisiológico: a interpretação é mudança em relação ao estado de atenção, não em relação à PAUSA.

É possível selecionar `erd_baseline_mode="preceding_pause"` (até 1,5 s na PAUSA anterior, margens de 250 ms) ou `"fixed"` (intervalo `baseline` do JSON, exigido dentro de PAUSA). Isso explica intervalos como −5 a −3,5 s: eram escolhidos na PAUSA anterior, não um deslocamento dos marcadores. A referência e os intervalos ficam explícitos no processamento/topografia; auditoria por trial em `baseline_audit.csv` e `erp_baseline_audit.csv`, sem tabela extensa na tela. Baseline inválida não é substituída silenciosamente por outra fase.

`active` define a janela de resumo/topografia; mu é [8,13) Hz e beta [13,30) Hz na grade TFR. Topografias indicam intervalo no ATTEMPT, bandas e referência; escalas são comuns e a cobertura limitada não equivale a uma digitalização individual.

## Features, classificadores, CV e teste online

TGSP = OAS → espaço tangente Riemann → StandardScaler. CSP usa regularização OAS → StandardScaler. Ambas podem alimentar **LDA**, **SVM linear** ou **SVM RBF**, identificados separadamente. Banco de bandas concatena features. Por padrão são comparados os três classificadores; o orçamento limita os candidatos por combinação features/classificador/protocolo.

Há resultados com papéis distintos:

1. **CV com parâmetros fixos:** cada candidato mantém janela, offset, bandas, features e classificador em todos os folds. Os mesmos conjuntos de trials são usados para todos os candidatos. Relatório geral mostra pontos por fold, boxplot quando há vários folds e média em losango. Janela fica no eixo x, offset nos painéis; bandas, features e hiperparâmetros distinguem condições sem linhas conectando-as. A chave comum aparece uma única vez em apêndice recolhido e em `parameter_combinations.csv`. A melhor média seleciona o modelo fixo de implantação. Esse resultado **é usado na seleção**; não é uma estimativa independente após escolher o melhor candidato.
2. **CV aninhada no IM_treino:** busca interna e avaliação externa separadas. O parâmetro escolhido pode variar por fold externo porque cada treino é diferente. Ela continua como estimativa separada da estratégia de seleção, não substitui a comparação de parâmetros fixos.
3. **Online:** uma parametrização é escolhida usando apenas IM_treino anterior, ajustada no treino e aplicada ao bloco online. Métricas online e timeline usam o mesmo modelo fixo e o mesmo processamento causal. A fonte de treino fica registrada. Sem `train_run_id`, usa-se a calibração incluída mais recente da mesma sessão/projeto/sujeito, anterior pelo timestamp. Como não se conhece necessariamente o modelo efetivamente usado na aquisição, esse teste continua identificado como **proxy de calibração anterior**.

Todos os splits e a calibração sigmoid separam **trials inteiros**. Janelas internas não são replicações independentes. Transformações aprendidas são ajustadas no treino. Candidatos precisam das classes e suporte temporal; falhas não são transformadas em resultados aprovados.

A timeline do modelo fixo processa o **sinal completo**, por janelas causais, inclusive antes/entre/depois das épocas; começa após a primeira janela completa, sem inventar suporte anterior. No IM_treino, a curva usa o modelo ajustado nesses dados e aparece como **in-sample, não validação**. Predições out-of-fold são exportadas separadamente. No online, a curva é de teste em bloco distinto. ATTEMPT é sombreado com classe e ID; as curvas alinhadas vêm dessa mesma timeline. Suavização causal é apenas visual; métricas e probabilidades exportadas não são suavizadas. Falhas de aquisição ainda exigem o check de timestamps; não se inventam amostras ausentes.

## PCA exploratório em teste

Usa vetores **tangentes TGSP**: covariâncias OAS → referência Riemann → scaler → PCA. A referência, o scaler e o PCA são ajustados **somente no treino**; dados de teste são projetados por essas mesmas transformações. Isso mantém uma representação reproduzível, em vez de ajustar uma redução nova no teste.

São usadas janelas reais de até 1 s, passo 250 ms, dentro de `active`. É uma expansão de observações dependentes, sem gerar pontos sintéticos e sem aumentar o número de trials independentes. No IM/EM, há um split estratificado por trial; no online, usa-se IM_treino anterior. Poucos trials ou falta de canais comuns produzem indisponibilidade explícita.

Hover informa classe, trial e início da janela; cores saturadas identificam as classes. Não há linhas ligando janelas nem KDE. A legenda isola os pontos, o centroide e a elipse da classe. Centroide e covariância dão o mesmo peso total a cada trial, independentemente da quantidade de janelas. A elipse de um desvio-padrão descreve dispersão; não é intervalo de confiança de observações independentes. Com menos de três trials ou covariância singular, a elipse é omitida. `pca_windows.csv` registra os papéis e IDs; `pca_model.json` registra transformações e origem.

## CLI e validação

```bash
python analyze.py index --root /caminho/dados
python analyze.py check --root /caminho/dados
python analyze.py batch --root /caminho/dados --mode both --protocols left_right reststim_move --compare-pipelines
python analyze.py report --root /caminho/dados --subjects AB01 --sessions S1
python -m pytest -q
```

`--config` aceita JSON parcial; filtros `--subjects`, `--sessions`, `--blocks` e `--runs` não alteram o controle salvo. Pela CLI, selecionar `--runs` distingue projetos quando o código do sujeito se repete.

Testes sintéticos cobrem múltiplos sujeitos, repetições, sessão EM/IM/online com perda manual de canal, três classificadores, folds idênticos por parametrização, separação por trial, baseline ATTENTION, mapa Italia, decimação e PCA em teste. Não substituem validação de unidades, montagem e limiares nas gravações reais.


## Revisão de legibilidade e indicadores

A aba de resultados mostra um dashboard por registro com épocas totais/aceitas/rejeitadas,
exclusões, alertas, disponibilidade e links individuais/de sessão. O snapshot permanece no
HTML individual: cinza/vermelho, escala física comum, mediana removida apenas visualmente,
offset de 1,3 × P99,5 em µV. Não há normalização por canal nem alteração das amplitudes.
Tabelas têm linhas compactas, números legíveis e detalhes longos nos CSV/JSON de auditoria.
Identificação de projeto usa BR/IT quando necessária no cabeçalho.

Timelines e curvas médias aparecem somente para **TGSP + SVM RBF**. LDA, SVM linear e CSP
continuam nas comparações e exportações. A coluna `selection_cv_bacc` é a média dos folds
fixos do candidato implantado (seleção); `balanced_accuracy` é a CV aninhada ou o teste
online, conforme `validation`. Online tem um ponto por resultado, sem distribuição fictícia.
Inferência causal, modelo de treino anterior, predições OOF e suavização visual são preservados.
Ausência de classe tem status curto, como “Indisponível: REST_STIM ausente”; exceções completas
ficam em `classification_failures.json`. Erros com classes disponíveis permanecem identificados
como erros de execução, sem recodificação de classes ou substituição de protocolo.

**Qualidade descritiva:** amplitude robusta por canal = IQR × 0,7413 em µV; desvio em relação
à mediana dos canais dividido por IQR × 0,7413 entre amplitudes. |desvio| > 5 sinaliza revisão,
inspirado no critério de desvio do PREP, sem reproduzir todo o pipeline PREP e sem significado
clínico. Dispersão entre canais degenerada torna o desvio não mensurável. Sinal quase constante
usa pico a pico abaixo do `flat_uv` já configurado. Diferenças fisiológicas, referência e a
cobertura central limitada também alteram amplitude: nenhum alerta exclui canais.
A potência estreita 50/60 ±1 Hz usa Welch na aquisição, antes de antialias/low-pass;
sem pelo menos 2 s e Nyquist acima de 61 Hz, o indicador conjunto é indisponível, nunca zero.
A tabela compacta e `channel_quality.csv` preservam unidades e contribuição nas rejeições;
não se cria um escore global de qualidade.

**Potência absoluta:** PSD Welch em µV²/Hz, Hann de 1 s (ou a duração disponível),
sobreposição 50%. Espectrograma STFT com a mesma densidade, Hann de 1 s e sobreposição
75%, alinhado a ATTEMPT, em subplots C3/Cz/C4 quando disponíveis. Média entre épocas
em potência linear antes de 10 log10, em dB re 1 µV²/Hz; mesma escala entre canais/classes.
São análises do EEG limpo aceito, sem normalização por baseline: não são ERD/ERS nem
“PSD Morlet”. A referência ATTENTION/PAUSA selecionada continua explícita em ERD/ERS.

**CSD/Laplaciano:** não foi acrescentado filtro espacial. O MNE implementa um Laplaciano
por splines esféricas no sinal, com posições válidas, esfera e regularização; não é suavização
de imagem. A derivada espacial depende da cobertura e dos vizinhos, sobretudo nas bordas.
Os 15 canais centrais do Brasil não oferecem cobertura suficiente para recomendar CSD
como padrão. Os até 53 canais Italia oferecem maior cobertura, mas exigem confirmar posições,
referência de aquisição, canais perdidos, ajuste da esfera e estabilidade em relação à
montagem/regularização antes de validar uma opção exploratória. Posições de template não
substituem digitalização individual. CSD mudaria unidades e precisaria de estimativa de potência
posterior e validação próprias; classificação e análise padrão ficam preservadas.

### Fontes efetivamente consultadas

As páginas web MNE/SciPy e o artigo PREP retornaram HTTP 403 no ambiente. Foram lidos:

- [Tutorial oficial MNE de canais ruins, fonte](https://github.com/mne-tools/mne-python/blob/main/tutorials/preprocessing/15_handling_bad_channels.py): marcação manual em `info['bads']` e inspeção sem apagar sinais.
- [Implementação/documentação oficial MNE CSD](https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/_csd.py): splines esféricas, esfera, posições, regularização, tratamento de canais ruins; referências Perrin e Kayser/Tenke citadas ali.
- [Implementação PyPREP de canais ruidosos](https://github.com/sappelhoff/pyprep/blob/main/pyprep/find_noisy_channels.py), `find_bad_by_deviation` e `find_bad_by_nan_flat`: IQR robusto, desvio e sinal quase constante. O artigo [PREP, Bigdely-Shamlo et al. (2015)](https://doi.org/10.3389/fninf.2015.00016) é referência de contexto, não uma leitura integral realizada nesta revisão.
- Docstrings instaladas de `scipy.signal.welch` e `scipy.signal.spectrogram` (SciPy 1.18.1): densidade, unidades e janelas; [página oficial Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

Validação usa dados sintéticos: múltiplos sujeitos, repetições, IM/online, classes ausentes,
janelas/offsets/bancos de bandas e modelos diferentes. Não há gravações reais neste checkout;
isso não valida fisiologia, unidades da aquisição ou adequação de limiares em dados reais.
