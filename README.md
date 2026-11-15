# Обнаружение аномалий в поведении серверных служб Linux с применением алгоритмов машинного обучения

Курсовой проект. Офлайн-анализ трасс системных вызовов набора данных LID-DS (2021):
основной метод GMM (модель гауссовых смесей), для сравнения Isolation Forest, One-Class SVM, LOF,
baseline STIDE и правило по длительности записи. Обучение только на нормальных данных (semi-supervised, one-class).

Сценарии: Bruteforce CWE-307, CVE-2012-2122 (MySQL), CVE-2014-0160 (Heartbleed).

## Структура проекта

```
data/            наборы LID-DS (не хранятся в репозитории)
  Bruteforce_CWE-307/Bruteforce_CWE-307/{training,validation,test}
  CVE-2012-2122/{training,validation,test}
  CVE-2014-0160/{training,validation,test}
exploration/     скрипты первичного изучения данных (структура, метки, время эксплойта)
src/             основной код
results/         результаты экспериментов (CSV, графики, windows_*.npz)
models/          сохранённые модели (.joblib)
```

## Установка

```
python -m venv .venv
.venv\Scripts\activate          # Windows; в Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Данные LID-DS нужно скачать отдельно (проект LID-DS: https://github.com/LID-DS/LID-DS)
и распаковать так, чтобы внутри каждого сценария были папки `training`, `validation`, `test`.

## Порядок запуска

Все команды выполняются из корня проекта. Имена папок сценариев подставьте свои.

**1. Признаки и эксперимент для разных размеров окна, часть dev** (выбор размера окна и агрегатора):

```
python src/sweep.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 bruteforce 20 100 400 --part dev --seeds 3
python src/sweep.py data/CVE-2012-2122 cve_2012_2122 20 100 400 --part dev --seeds 3
python src/sweep.py data/CVE-2014-0160 cve_2014_0160 20 100 400 --part dev --seeds 3
```

Вывод: `results/<сценарий>_W<W>/` (`windows_*.npz`, `metrics.csv`, `roc_window.png`) и `results/sweep_<сценарий>_dev.csv`.
Выбранная конфигурация: W = 400, агрегатор записи p95.

**2. Исследование GMM и проверка положения окна, часть dev:**

```
python src/gmm_study.py results/bruteforce_W400
python src/gmm_study.py results/cve_2012_2122_W400
python src/position_check.py results/bruteforce_W400 --k 16 --cov full
python src/position_check.py results/cve_2012_2122_W400 --k 8 --cov full
```

**3. Итоговая оценка, часть final. Запускать один раз после фиксации конфигурации:**

```
python src/sweep.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 bruteforce 400 --part final --seeds 5
python src/sweep.py data/CVE-2012-2122 cve_2012_2122 400 --part final --seeds 5
python src/sweep.py data/CVE-2014-0160 cve_2014_0160 400 --part final --seeds 5
```

**4. Baseline (STIDE и правило по длительности записи):**

```
python src/baselines.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 results/bruteforce_W400 --part dev
python src/baselines.py data/Bruteforce_CWE-307/Bruteforce_CWE-307 results/bruteforce_W400 --part final
python src/baselines.py data/CVE-2012-2122 results/cve_2012_2122_W400 --part dev
python src/baselines.py data/CVE-2012-2122 results/cve_2012_2122_W400 --part final
```

**5. Обучение и сохранение итоговой GMM, разбор записей и график:**

```
python src/train_final.py results/bruteforce_W400 models/bruteforce_gmm.joblib
python src/detect.py models/bruteforce_gmm.joblib "data/Bruteforce_CWE-307/Bruteforce_CWE-307/test/normal" --limit 3
python src/detect.py models/bruteforce_gmm.joblib "data/Bruteforce_CWE-307/Bruteforce_CWE-307/test/normal_and_attack" --limit 3
python src/plot_record.py models/bruteforce_gmm.joblib НОРМАЛЬНАЯ.zip АТАКУЮЩАЯ.zip results/bruteforce_record_scores.png
```

## Описание скриптов `src/`

| Файл | Назначение |
|---|---|
| `features_windows.py` | чтение записей, словарь, окна, частотные признаки, разметка по времени эксплойта, `windows_*.npz` |
| `run_experiment.py` | обучение четырёх моделей, пороги, метрики на уровне окон и записей, несколько seed |
| `sweep.py` | запуск двух предыдущих для набора размеров окна |
| `gmm_study.py` | сетка числа компонент и ковариации, чувствительность к порогу, компоненты, вклад вызовов |
| `position_check.py` | проверка, не объясняется ли результат положением окна в записи |
| `baselines.py` | STIDE и правило по длительности записи |
| `train_final.py` | обучение и сохранение итоговой GMM с порогами |
| `detect.py` | разбор записей сохранённой моделью |
| `plot_record.py` | график оценки окон во времени |

## Воспроизводимость

- Разбиение test на dev и final: 50/50 по записям, стратификация по наличию атаки, фиксированный seed 12345.
- Модели: несколько seed (0..N-1); One-Class SVM и LOF детерминированы, их разброс равен нулю.
- Словарь вызовов, масштабирование и все пороги определяются только по training и validation.
- Длительность записи в признаки моделей не входит (иначе возникает утечка: нормальные записи длятся 40-44 с, атакующие 10-29 с).

## Ограничения

- Офлайн-анализ записанных трасс, не мониторинг в реальном времени.
- Результаты относятся к трём сценариям LID-DS и не переносятся на другие сервисы без проверки.
- Heartbleed (CVE-2014-0160) частотными признаками не обнаруживается.
- Записи короче окна не анализируются.
- Время окончания эксплойта в метаданных LID-DS отсутствует, разметка окон после начала атаки приблизительная.
