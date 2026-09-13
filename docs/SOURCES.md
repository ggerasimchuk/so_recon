# SOURCES.md — основания согласованной архитектуры SO-RECON 4.0

**Дата проверки доступных первоисточников:** 13 сентября 2026 г. Реестр сопоставляет ссылки Rxx в SPEC/COMPUTE_PROFILES с конкретными источниками. Он составлен для текущей редакции; отсутствующий исходный SOURCES из присланного комплекта не выдаётся за восстановленный документ.

Статус проверки различается: доступные страницы/аннотации подтверждают библиографию и указанный ограниченный тезис; документация/код — обозначенные API-свойства. Это не полный аудит всех статей, не проверка результатов авторов на нашем поле и не обещание совместимости latest API с lockfile. Перед реализацией примеры сверяются с закреплёнными Julia/Jutul/JutulDarcy версиями; смена upstream stable не обновляет проект автоматически.

## Методы и физический backend

<a id="r01"></a>
**R01.** Møyner O. *JutulDarcy.jl — a fully differentiable high-performance reservoir simulator based on automatic differentiation*. Computational Geosciences, 2025. [Статья и DOI](https://link.springer.com/article/10.1007/s10596-025-10366-6), [официальное описание проекта](https://sintefmath.github.io/JutulDarcy.jl/stable/index). Подтверждены библиографическая запись и архитектурная направленность; выбор backend не доказывает точность конкретной полевой модели.

<a id="r02"></a>
**R02.** JutulDarcy, [Wells and controls](https://sintefmath.github.io/JutulDarcy.jl/stable/man/basics/wells). Документация различает liquid/water/total targets и отключение поверхности от закрытия перфораций. `DisabledControl` допускает внутренний обмен при открытых соединениях; конкретное поведение закреплённой версии проверяется E01 crossflow tests.

<a id="r03"></a>
**R03.** JutulDarcy, [Supported physical systems](https://sintefmath.github.io/JutulDarcy.jl/stable/man/basics/systems). Документированы immiscible и black-oil classes. Наличие класса в библиотеке не задаёт пригодный PVT prior месторождения.

<a id="r04"></a>
**R04.** Durkan C., Bekasov A., Murray I., Papamakarios G. *Neural Spline Flows*. NeurIPS, 2019. [Авторская публикация](https://arxiv.org/abs/1906.04032). Рационально-квадратичные сплайны, обратимость и вычисление density. Основание выбора NSF, без утверждения превосходства в нашей inverse-задаче.

<a id="r05"></a>
**R05.** Papamakarios G. et al. *Normalizing Flows for Probabilistic Modeling and Inference*. JMLR 22(57):1–64, 2021. [Статья](https://jmlr.org/papers/v22/19-1028.html). Общая change-of-variables постановка, выразительность и вычислительные компромиссы flows.

<a id="r06"></a>
**R06.** Wildberger J. et al. *Flow Matching for Scalable Simulation-Based Inference*. NeurIPS, 2023. [Публикация](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html). FMPE также поддерживает density evaluation; NSF выбран как исходное инженерное решение, а не вследствие невозможности плотности у FMPE.

<a id="r08"></a>
**R08.** Greenberg D., Nonnenmacher M., Macke J. *Automatic Posterior Transformation for Likelihood-Free Inference*. ICML/PMLR 97, 2019. [Статья](https://proceedings.mlr.press/v97/greenberg19a.html). Основание учитывать distribution shift при sequential simulation proposals. Выбранная в SPEC importance-weighted NLL — отдельное решение проекта, **не реализация APT** и не приписываемая авторам формула.

<a id="r10"></a>
**R10.** Cui T., Martin J., Marzouk Y. M., Solonen A., Spantini A. *Likelihood-informed dimension reduction for nonlinear inverse problems*. [Авторская публикация](https://arxiv.org/abs/1403.4680), 2014. Основание исследовать информативные направления; готовый LIS и его полезность для данного поля не предполагаются.

<a id="r12"></a>
**R12.** Talts S., Betancourt M., Simpson D., Vehtari A., Gelman A. *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. [Авторская публикация](https://arxiv.org/abs/1804.06788), 2018. SBC проверяет inference под заданным generative model; не подтверждает физическую истинность самого генератора.

<a id="r13"></a>
**R13.** Emerick A. A., Reynolds A. C. *Ensemble smoother with multiple data assimilation*. Computers & Geosciences, 2013. [DOI](https://doi.org/10.1016/j.cageo.2012.03.011). Библиографическая запись сохранена из SPEC 3.0/RESEARCH; прямое открытие издательской страницы в этой проверке не удалось. ES-MDA используется как сильный ансамблевый baseline; его соответствие конкретной observation model проверяется отдельным experiment, не предполагается для bounded Student-t.

## Ресурсы и интерфейсы

<a id="r27"></a>
**R27.** PyTorch, [MPS OperationUtils source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.mm). Проверен отказ tensor conversion для Float64. Training MPS остаётся опцией после проверки выбранных операций; operational proposal/weights по умолчанию CPU Float64.

<a id="r28"></a>
**R28.** JutulDarcy, [GPU support](https://sintefmath.github.io/JutulDarcy.jl/stable/man/advanced/gpu). Описанное экспериментальное ускорение linear solves относится к CUDA, с отдельными ограничениями AMGX. Этот источник не обосновывает Apple GPU backend для solver.

<a id="r29"></a>
**R29.** PyTorch, [set_per_process_memory_fraction](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_per_process_memory_fraction.html). Открытая stable-ссылка перенаправила на эту версию документации. Лимит связан с Metal recommended working-set size; 0 означает unlimited. Бюджет 12/16 GiB в проекте не взят из этой функции и контролируется отдельно.

<a id="r30"></a>
**R30.** Julia, [Platform Specific Instructions](https://julialang.org/downloads/platform/). Основание проверки установки на macOS/Apple Silicon. Воспроизводимая версия проекта остаётся указанной в `julia/.julia-version`; latest release не является её заменой.

<a id="r31"></a>
**R31.** JutulDarcy, [Adjoints and gradients](https://sintefmath.github.io/JutulDarcy.jl/stable/man/basics/adjoints). Документация интерфейсов чувствительности. Конкретный chain-rule objective для нашей normalized monthly likelihood — проектный контракт; его реализация и finite-difference gate ещё предстоят в E10.

<a id="r32"></a>
**R32.** JutulDarcy, [High-level API](https://sintefmath.github.io/JutulDarcy.jl/stable/man/highlevel). `extra_outputs=false` описан как минимальный набор сохраняемых переменных; это не самостоятельная гарантия streaming или постоянной памяти по длине истории. Restart/retained states проверяются в E01/E06.

## Проектные решения и локальные свидетельства

Размеры сеток/корпуса/сети, выбранный bridge r→p0·L, состав bounded-bin likelihood, epsilon, CESS thresholds, RAM/wall limits и C_* acceptance thresholds являются **проектными решениями**. Их корректность/польза проверяется по SPEC, а численные настройки фиксируются до закрытого теста. Ни одна статья не объявляется источником гарантии «10% улучшения So», полного posterior на 32 частицах или исследования за 24 часа.

- [DATA_AUDIT.md](DATA_AUDIT.md) — сохранённый аудит и ограничения семантики CSV; E04 воспроизводит вычислимые утверждения.
- [Source manifest E00](../reports/manifests/source_manifest.json) — hashes/размеры/физические строки локальных исходников.
- [Отчёт E00](../reports/stages/E00.md) — свидетельства foundation по 3.0, не результаты inverse pipeline.
- [RESEARCH.md](RESEARCH.md) — прежний литературный синтез; трактовки архитектуры обновлены решениями [DECISIONS.md](DECISIONS.md).

Подтверждённые секторные PVT/ОФП/Sorw и временные замеры давления этим реестром не предоставлены. Они входят в external_priors ledger будущего E05 с единицами, датой/глубиной, источником и обоснованием переноса. До их получения учебные synthetic параметры не подписываются как свойства месторождения.
