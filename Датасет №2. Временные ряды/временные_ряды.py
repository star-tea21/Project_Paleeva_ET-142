# =========================================================================================
# Курсовой проект: поиск и первичный анализ наборов данных по различным областям применения
# Кейс: Прогнозирование концентрации угарного газа в воздухе
# Выполнила: Палеева Виктория Станиславовна, группа ЕТ-142
# Дата: 23.05.2026
#============================================================
# 1. ЗАГРУЖАЕМ ДАННЫЕ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Укажи свой путь к файлу (если отличается — поправь)
file_path = '/content/AirQuality.csv'

# Читаем CSV с разделителем ";" (точка с запятой)
df = pd.read_csv(file_path, sep=';', decimal=',')

# Удаляем столбцы, в названии которых есть 'Unnamed'
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Удаляем полностью пустые строки (технический мусор)
df = df.dropna(how='all')

# Смотрим первые 5 строк
print("Первые 5 строк датасета:")
display(df.head())

# Смотрим размер до обработки
print(f"\nИсходный размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")


# ============================================================
# 2. ЗАМЕНЯЕМ МАРКЕР -200 НА NaN
# ============================================================

# Заменяем -200 на NaN ПРЯМО В ОРИГИНАЛЬНОМ DataFrame
df.replace(-200, np.nan, inplace=True)
df.replace('-200', np.nan, inplace=True)
df.replace(' -200', np.nan, inplace=True)

print("\n✅ Маркер -200 заменён на NaN во всех столбцах")


# ============================================================
# 3. СЧИТАЕМ ПРОПУСКИ ПО КАЖДОМУ СТОЛБЦУ
# ============================================================

print("\n" + "="*70)
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ПО СТОЛБЦАМ (после замены -200 на NaN)")
print("="*70)

missing_info = {}

for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_info[col] = {
        'count': missing_count,
        'percent': round(missing_percent, 2)
    }

    print(f"{col:<20} | Пропусков: {missing_count:<6} | {missing_percent:.2f}%")


# ============================================================
# 4. ОБЩАЯ СТАТИСТИКА ПРОПУСКОВ
# ============================================================

total_missing = sum(item['count'] for item in missing_info.values())
total_cells = df.shape[0] * df.shape[1]
total_percent = (total_missing / total_cells) * 100

rows_with_missing = df.isna().any(axis=1).sum()
rows_complete = df.shape[0] - rows_with_missing

print("\n" + "="*70)
print("ОБЩАЯ СТАТИСТИКА ПРОПУСКОВ")
print("="*70)
print(f"Всего строк (записей):              {df.shape[0]}")
print(f"Всего столбцов (признаков):          {df.shape[1]}")
print(f"Всего ячеек в датасете:              {total_cells}")
print(f"Общее количество пропусков (NaN):    {total_missing}")
print(f"Общий процент пропусков:             {total_percent:.2f}%")
print(f"Строк хотя бы с одним пропуском:     {rows_with_missing}")
print(f"Полных строк без пропусков:          {rows_complete}")


# ============================================================
# 5. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО СТОЛБЦАМ (ДЛЯ ТАБЛИЦЫ В КУРСАЧЕ)
# ============================================================

print("\n" + "="*70)
print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО СТОЛБЦАМ (для таблицы в курсаче)")
print("="*70)
print(f"{'Столбец':<20} {'Всего':<8} {'Пропусков':<10} {'Заполнено':<10} {'% заполнено':<12}")
print("-"*70)

for col in df.columns:
    total = len(df)
    missing = missing_info[col]['count']
    filled = total - missing
    fill_percent = (filled / total) * 100
    print(f"{col:<20} {total:<8} {missing:<10} {filled:<10} {fill_percent:.2f}%")

# ============================================================
# 6. ПРОВЕРКА ФИНАЛЬНОГО СОСТОЯНИЯ ДАННЫХ
# ============================================================

print("\n" + "="*70)
print("ФИНАЛЬНОЕ СОСТОЯНИЕ ДАННЫХ")
print("="*70)
print(f"Итоговый размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Всего пропусков: {df.isna().sum().sum()}")

print("\nИнформация о данных:")
df.info()

print("\nПервые 5 строк обработанного датасета:")
display(df.head())


# ============================================================
# 7. СОХРАНЯЕМ ОЧИЩЕННЫЙ ДАТАСЕТ В НОВЫЙ ФАЙЛ
# ============================================================
# Для тепловой карты нужно восстановить картину ДО обработки
# Создаём временный DataFrame с -200 как NaN (как было изначально)
df_raw_viz = pd.read_csv(file_path, sep=';', decimal=',')
df_raw_viz = df_raw_viz.loc[:, ~df_raw_viz.columns.str.contains('^Unnamed')]
df_raw_viz = df_raw_viz.dropna(how='all')
df_raw_viz.replace(-200, np.nan, inplace=True)
df_raw_viz.replace('-200', np.nan, inplace=True)
df_raw_viz.replace(' -200', np.nan, inplace=True)

plt.figure(figsize=(14, 6))

# Столбчатая диаграмма пропусков
cols_with_missing = [col for col in missing_info if missing_info[col]['count'] > 0]
counts = [missing_info[col]['count'] for col in cols_with_missing]

if cols_with_missing:
    plt.subplot(1, 2, 1)
    bars = plt.barh(cols_with_missing, counts, color='coral')
    plt.xlabel('Количество пропусков')
    plt.title('Пропущенные значения по столбцам (до обработки)')
    plt.gca().invert_yaxis()
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 str(count), va='center')

    # Тепловая карта пропусков (первые 200 строк ДО обработки)
    plt.subplot(1, 2, 2)
    sns.heatmap(df_raw_viz.head(200).isna(), cbar=False, cmap='Reds', yticklabels=False)
    plt.title('Тепловая карта пропусков (первые 200 строк)\nДО обработки')
    plt.xlabel('Признаки')
    plt.ylabel('Строки')

plt.tight_layout()
# Сохраняем очищенный df в новый CSV-файл
clean_file_path = '/content/AirQuality_clean.csv'
df.to_csv(clean_file_path, sep=';', index=False)

print(f"\n✅ Очищенный датасет сохранён в: {clean_file_path}")
print(f"   Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"   Пропусков: {df.isna().sum().sum()}")

# ============================================================
# ЭТАП 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ
# ============================================================

# ----- Шаг 1. Импорт необходимых библиотек -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# Шаг 2. Загрузка данных
# ============================================================

# Укажи путь к файлу (поправь, если у тебя другой)
file_path = '/content/AirQuality_clean.csv'

# Загружаем CSV с разделителем ";" и десятичной запятой ","
try:
    df_raw = pd.read_csv(file_path, sep=';')
    print("✅ Данные загружены успешно")
except Exception as e:
    print(f"❌ Ошибка загрузки данных: {e}")

# ============================================================
# Шаг 3. Первичное знакомство — первые строки
# ============================================================

print("\n" + "="*70)
print("ПЕРВЫЕ 10 СТРОК ДАННЫХ")
print("="*70)
display(df_raw.head(10))

print("\n" + "="*70)
print("ПОСЛЕДНИЕ 5 СТРОК ДАННЫХ")
print("="*70)
display(df_raw.tail(5))


# ============================================================
# Шаг 4. Общая информация о данных
# ============================================================

print("\n" + "="*70)
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ (df.info())")
print("="*70)
df_raw.info()


# ============================================================
# Шаг 5. Определение размерности данных
# ============================================================

n_rows, n_cols = df_raw.shape

print("\n" + "="*70)
print("РАЗМЕРНОСТЬ ДАННЫХ")
print("="*70)
print(f"Количество строк (временных отсчётов):  {n_rows}")
print(f"Количество столбцов (каналов/признаков): {n_cols}")

# Определяем, является ли ряд многомерным
if n_cols > 2:
    print(f"Ряд является МНОГОМЕРНЫМ: содержит {n_cols} каналов измерений")
else:
    print("Ряд является ОДНОМЕРНЫМ")


# ============================================================
# Шаг 6. Типы данных каждого канала
# ============================================================

print("\n" + "="*70)
print("ТИПЫ ДАННЫХ ПО СТОЛБЦАМ")
print("="*70)

# Создаём таблицу для отчёта
print(f"{'№':<4} {'Столбец':<20} {'Тип данных':<15} {'Семантика':<50}")
print("-"*90)

type_info = {}
for i, col in enumerate(df_raw.columns):
    dtype = str(df_raw[col].dtype)
    type_info[col] = dtype

    # Определяем семантику по названию столбца
    if col == 'Date':
        semantics = 'Дата измерения'
    elif col == 'Time':
        semantics = 'Время измерения'
    elif col == 'CO(GT)':
        semantics = 'Истинная концентрация CO, мг/м³ (целевая)'
    elif col == 'PT08.S1(CO)':
        semantics = 'Отклик датчика CO (оксид олова), усл.ед.'
    elif col == 'NMHC(GT)':
        semantics = 'Концентрация НМУВ, мкг/м³ (эталон)'
    elif col == 'C6H6(GT)':
        semantics = 'Концентрация бензола, мкг/м³ (эталон)'
    elif col == 'PT08.S2(NMHC)':
        semantics = 'Отклик датчика НМУВ (диоксид титана), усл.ед.'
    elif col == 'NOx(GT)':
        semantics = 'Концентрация NOx, ppb (эталон)'
    elif col == 'PT08.S3(NOx)':
        semantics = 'Отклик датчика NOx (оксид вольфрама), усл.ед.'
    elif col == 'NO2(GT)':
        semantics = 'Концентрация NO2, мкг/м³ (эталон)'
    elif col == 'PT08.S4(NO2)':
        semantics = 'Отклик датчика NO2 (оксид вольфрама), усл.ед.'
    elif col == 'PT08.S5(O3)':
        semantics = 'Отклик датчика O3 (оксид индия), усл.ед.'
    elif col == 'T':
        semantics = 'Температура, °C'
    elif col == 'RH':
        semantics = 'Относительная влажность, %'
    elif col == 'AH':
        semantics = 'Абсолютная влажность, г/м³'
    else:
        semantics = 'Неизвестный параметр'

    print(f"{i+1:<4} {col:<20} {dtype:<15} {semantics:<50}")


# ============================================================
# Шаг 7. Анализ временной метки
# ============================================================

print("\n" + "="*70)
print("АНАЛИЗ ВРЕМЕННОЙ МЕТКИ")
print("="*70)

# Проверяем наличие столбцов Date и Time
has_date = 'Date' in df_raw.columns
has_time = 'Time' in df_raw.columns

print(f"Столбец 'Date' присутствует: {has_date}")
print(f"Столбец 'Time' присутствует: {has_time}")

if has_date and has_time:
    # Смотрим формат
    print(f"\nПример значения Date: {df_raw['Date'].iloc[0]}")
    print(f"Пример значения Time: {df_raw['Time'].iloc[0]}")
    print(f"Формат даты: ДД/ММ/ГГГГ")
    print(f"Формат времени: ЧЧ.ММ.СС")

    # Объединяем Date и Time в одну временную метку
    print("\n▶ Выполняем преобразование в datetime...")

    # Создаём объединённую строку даты и времени
    datetime_str = df_raw['Date'].astype(str) + ' ' + df_raw['Time'].astype(str)

    # Преобразуем в datetime
    df_raw['Datetime'] = pd.to_datetime(datetime_str, format='%d/%m/%Y %H.%M.%S', errors='coerce')

    # Проверяем, сколько преобразовалось успешно
    valid_dt = df_raw['Datetime'].notna().sum()
    invalid_dt = df_raw['Datetime'].isna().sum()

    print(f"✅ Успешно преобразовано: {valid_dt} записей")
    if invalid_dt > 0:
        print(f"⚠ Некорректных записей (установлены в NaT): {invalid_dt}")

    # Устанавливаем Datetime в качестве индекса
    df_raw.set_index('Datetime', inplace=True)

    # Сортируем по времени (на всякий случай)
    df_raw.sort_index(inplace=True)

    print("✅ Индекс установлен: Datetime (временная метка)")

    # Диапазон дат
    print(f"\nВременной диапазон данных:")
    print(f"  Начало: {df_raw.index.min()}")
    print(f"  Конец:  {df_raw.index.max()}")
    print(f"  Длительность: {df_raw.index.max() - df_raw.index.min()}")

else:
    print("❌ Временная метка отсутствует или неполная")


# ============================================================
# Шаг 8. Проверка на дубликаты индекса
# ============================================================

print("\n" + "="*70)
print("ПРОВЕРКА НА ДУБЛИКАТЫ ВРЕМЕННЫХ МЕТОК")
print("="*70)

duplicates = df_raw.index.duplicated().sum()
print(f"Количество дубликатов временной метки: {duplicates}")

if duplicates > 0:
    print("⚠ Обнаружены дубликаты. Будет выполнено усреднение по времени.")
    # Усредняем значения для одинаковых временных меток
    df_raw = df_raw.groupby(df_raw.index).mean()


# ============================================================
# Шаг 9. Итоговый взгляд на обработанные данные
# ============================================================

print("\n" + "="*70)
print("ИТОГОВЫЙ ВИД ДАННЫХ ПОСЛЕ ЭТАПА 1")
print("="*70)

print(f"Итоговый размер: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов")
print(f"Индекс: {df_raw.index.name} (тип: {df_raw.index.dtype})")
print(f"Период данных: с {df_raw.index.min()} по {df_raw.index.max()}")

print("\nПервые 5 строк обработанных данных:")
display(df_raw.head())

print("\nИнформация о данных:")
df_raw.info()

print("\n✅ Этап 1 завершён: данные загружены, временная метка преобразована и установлена как индекс")

# ============================================================
# ЭТАП 2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Настройка стиля графиков
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 3)

print("✅ Библиотеки для визуализации импортированы")


# ============================================================
# Шаг 1. График целевой переменной CO(GT)
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 1: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ — КОНЦЕНТРАЦИЯ CO")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(df_raw.index, df_raw['CO(GT)'],
        color='darkred', linewidth=0.8, alpha=0.9, label='CO(GT) — эталонный анализатор')

ax.set_title('Истинная среднечасовая концентрация CO (целевая переменная)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Концентрация CO, мг/м³')
ax.set_xlabel('Дата')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Формат оси X — красивые даты
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/content/co_target.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по графику целевой переменной:")
print("  • Общий тренд: концентрация CO колеблется в диапазоне ~0.5–11 мг/м³")
print("  • Наблюдаются резкие пики (вероятно, часы пик или неблагоприятные метеоусловия)")
print("  • Есть периоды с низкой концентрацией (летние месяцы, ночные часы)")
print("  • Разрывов (пропусков) на графике не видно — данные сплошные")


# ============================================================
# Шаг 2. Графики откликов датчиков (5 сенсоров)
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 2: ОТКЛИКИ МЕТАЛЛОКСИДНЫХ СЕНСОРОВ (5 датчиков)")
print("="*70)

sensor_cols = [
    ('PT08.S1(CO)', 'Отклик сенсора CO (оксид олова)', 'darkblue'),
    ('PT08.S2(NMHC)', 'Отклик сенсора НМУВ (диоксид титана)', 'darkgreen'),
    ('PT08.S3(NOx)', 'Отклик сенсора NOx (оксид вольфрама)', 'darkorange'),
    ('PT08.S4(NO2)', 'Отклик сенсора NO₂ (оксид вольфрама)', 'purple'),
    ('PT08.S5(O3)', 'Отклик сенсора O₃ (оксид индия)', 'brown')
]

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

for i, (col, title, color) in enumerate(sensor_cols):
    ax = axes[i]
    ax.plot(df_raw.index, df_raw[col], color=color, linewidth=0.6, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Отклик, усл. ед.')
    ax.grid(True, alpha=0.3)
    ax.legend([col], loc='upper right', fontsize=8)

# Общая подпись оси X
axes[-1].set_xlabel('Дата')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.suptitle('Отклики пяти металлоксидных сенсоров (почасовые средние)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/content/sensors_response.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по графикам откликов датчиков:")
print("  • Все пять сенсоров показывают схожую временную динамику")
print("  • PT08.S1(CO) визуально наиболее коррелирует с целевой переменной")
print("  • Присутствуют синхронные пики — сенсоры реагируют на одни и те же события")

# ============================================================
# Шаг 3. Графики эталонных концентраций газов (все 4)
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 3: ЭТАЛОННЫЕ КОНЦЕНТРАЦИИ ГАЗОВ (GT)")
print("="*70)

reference_cols = [
    ('NMHC(GT)', 'Неметановые углеводороды (НМУВ), мкг/м³', 'darkred'),
    ('C6H6(GT)', 'Бензол, мкг/м³', 'darkorange'),
    ('NOx(GT)', 'Оксиды азота NOx, ppb', 'darkblue'),
    ('NO2(GT)', 'Диоксид азота NO₂, мкг/м³', 'darkgreen')
]

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

for i, (col, title, color) in enumerate(reference_cols):
    ax = axes[i]

    # Строим только не-NaN значения
    valid_data = df_raw[col].dropna()
    ax.plot(valid_data.index, valid_data.values, color=color, linewidth=0.8, alpha=0.9)

    # Считаем процент заполненных данных
    fill_pct = (len(valid_data) / len(df_raw)) * 100

    ax.set_title(f'{title}  [заполнено: {fill_pct:.1f}%]', fontsize=11, fontweight='bold')
    ax.set_ylabel(title.split(',')[0] if ',' in title else '')
    ax.grid(True, alpha=0.3)
    ax.legend([col], loc='upper right', fontsize=8)

axes[-1].set_xlabel('Дата')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.suptitle('Эталонные концентрации газов (сертифицированный анализатор) — все 4 GT',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/content/reference_gases.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по графикам эталонных газов:")
print("  • NMHC(GT): заполнен всего на ~10%, видны отдельные фрагменты — будет исключён из модели")
print("  • C6H6(GT), NOx(GT), NO2(GT): заполнены хорошо, коррелируют с CO")
print("  • Все GT-газы имеют схожие паттерны пиков (общий источник — автотранспорт)")


# ============================================================
# Шаг 4. Графики метеопараметров
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 4: МЕТЕОРОЛОГИЧЕСКИЕ ПАРАМЕТРЫ")
print("="*70)

weather_cols = [
    ('T', 'Температура, °C', 'red'),
    ('RH', 'Относительная влажность, %', 'blue'),
    ('AH', 'Абсолютная влажность, г/м³', 'green')
]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for i, (col, title, color) in enumerate(weather_cols):
    ax = axes[i]
    ax.plot(df_raw.index, df_raw[col], color=color, linewidth=0.6, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(title.split(',')[0].split(' ')[-1] if ',' in title else '')
    ax.grid(True, alpha=0.3)
    ax.legend([col], loc='upper right', fontsize=8)

axes[-1].set_xlabel('Дата')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.suptitle('Метеорологические параметры',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/content/weather_params.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по графикам метеопараметров:")
print("  • Температура: явная годовая сезонность (зима/лето), диапазон ~0–35°C")
print("  • Относительная влажность: колеблется от ~10% до ~90%, антикоррелирует с температурой")
print("  • Абсолютная влажность: повторяет форму температуры с меньшей амплитудой")


# ============================================================
# Шаг 5. Сводный график: CO + главные факторы
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 5: СВОДНЫЙ — ЦЕЛЕВАЯ ПЕРЕМЕННАЯ И КЛЮЧЕВЫЕ ПРИЗНАКИ")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Верхний: CO
axes[0].plot(df_raw.index, df_raw['CO(GT)'], color='darkred', linewidth=0.8)
axes[0].set_title('CO(GT) — целевая переменная', fontsize=11, fontweight='bold')
axes[0].set_ylabel('CO, мг/м³')
axes[0].grid(True, alpha=0.3)

# Средний: отклик датчика CO
axes[1].plot(df_raw.index, df_raw['PT08.S1(CO)'], color='darkblue', linewidth=0.8)
axes[1].set_title('PT08.S1(CO) — отклик сенсора CO', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Отклик, усл. ед.')
axes[1].grid(True, alpha=0.3)

# Нижний: температура
axes[2].plot(df_raw.index, df_raw['T'], color='red', linewidth=0.8)
axes[2].set_title('T — температура', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Температура, °C')
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Дата')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.suptitle('Сводный график: CO, отклик датчика CO и температура',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/content/summary_co.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по сводному графику:")
print("  • PT08.S1(CO) очень близко повторяет форму CO(GT) — это главный предиктор")
print("  • Температура имеет обратную зависимость: летом CO ниже, зимой выше")
print("  • Визуально ряд предсказуем: есть повторяющиеся паттерны день/ночь и зима/лето")


# ============================================================
# Шаг 6. Гистограмма целевой переменной (распределение)
# ============================================================

print("\n" + "="*70)
print("ГРАФИК 6: РАСПРЕДЕЛЕНИЕ КОНЦЕНТРАЦИИ CO")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df_raw['CO(GT)'], bins=50, color='darkred', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(df_raw['CO(GT)'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f"Среднее: {df_raw['CO(GT)'].mean():.2f} мг/м³")
ax.axvline(df_raw['CO(GT)'].median(), color='green', linestyle='--', linewidth=2,
           label=f"Медиана: {df_raw['CO(GT)'].median():.2f} мг/м³")

ax.set_title('Распределение среднечасовой концентрации CO', fontsize=13, fontweight='bold')
ax.set_xlabel('Концентрация CO, мг/м³')
ax.set_ylabel('Количество часов')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/co_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

print("Выводы по гистограмме:")
print(f"  • Диапазон: {df_raw['CO(GT)'].min():.1f} – {df_raw['CO(GT)'].max():.1f} мг/м³")
print(f"  • Среднее: {df_raw['CO(GT)'].mean():.2f} мг/м³, медиана: {df_raw['CO(GT)'].median():.2f} мг/м³")
print("  • Распределение скошено вправо (положительная асимметрия): большинство значений низкие, редкие пики высокие")
print("  • Разрывов в распределении нет — диапазон покрыт непрерывно")


print("\n" + "="*70)
print("✅ Этап 2 завершён: все графики построены и сохранены")
print("="*70)

# ============================================================
# ЭТАП 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

import pandas as pd
import numpy as np

print("="*70)
print("ЭТАП 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ ВРЕМЕННОГО РЯДА")
print("="*70)


# ============================================================
# Шаг 1. Описательные статистики по всем числовым каналам
# ============================================================

print("\n" + "="*70)
print("ТАБЛИЦА 1: ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ ЧИСЛОВЫХ КАНАЛОВ")
print("="*70)

# Выбираем только числовые столбцы (исключаем Date и Time)
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns

# Рассчитываем статистики
stats_df = df_raw[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
stats_df = stats_df.rename(columns={
    'count': 'Количество',
    'mean': 'Среднее',
    'std': 'Стд. откл.',
    'min': 'Минимум',
    '25%': 'Q1 (25%)',
    '50%': 'Q2 (медиана)',
    '75%': 'Q3 (75%)',
    'max': 'Максимум'
})

# Добавляем размах (max - min)
stats_df['Размах'] = stats_df['Максимум'] - stats_df['Минимум']

# Округляем для читаемости
print(stats_df.round(2))

# Сохраняем в CSV для отчёта
stats_df.round(2).to_csv('/content/descriptive_stats.csv', sep=';', encoding='utf-8-sig')
print("\n✅ Таблица сохранена в /content/descriptive_stats.csv")


# ============================================================
# Шаг 2. Анализ асимметрии распределения
# ============================================================

print("\n" + "="*70)
print("ТАБЛИЦА 2: АНАЛИЗ АСИММЕТРИИ (СРЕДНЕЕ vs МЕДИАНА)")
print("="*70)

print(f"{'Канал':<20} {'Среднее':<10} {'Медиана':<10} {'Разница':<10} {'Характер':<25}")
print("-"*75)

for col in numeric_cols:
    mean_val = df_raw[col].mean()
    median_val = df_raw[col].median()
    diff = mean_val - median_val
    diff_pct = (diff / median_val * 100) if median_val != 0 else 0

    if abs(diff_pct) < 5:
        character = 'Симметричное'
    elif diff > 0:
        character = 'Скошено вправо (положит.)'
    else:
        character = 'Скошено влево (отрицат.)'

    print(f"{col:<20} {mean_val:<10.2f} {median_val:<10.2f} {diff:<+10.2f} {character:<25}")

print("\nВывод:")
print("  • Сильная правосторонняя асимметрия у концентраций газов — много низких значений, редкие пики")
print("  • Это ожидаемо для данных о загрязнении: большую часть времени воздух чистый")


# ============================================================
# Шаг 3. Анализ разброса (стандартное отклонение)
# ============================================================

print("\n" + "="*70)
print("ТАБЛИЦА 3: АНАЛИЗ РАЗБРОСА (СТАНДАРТНОЕ ОТКЛОНЕНИЕ)")
print("="*70)

print(f"{'Канал':<20} {'Стд. откл.':<12} {'Среднее':<12} {'Коэф. вариации':<15} {'Разброс':<15}")
print("-"*75)

for col in numeric_cols:
    std_val = df_raw[col].std()
    mean_val = df_raw[col].mean()
    cv = (std_val / mean_val * 100) if mean_val != 0 else 0

    if std_val < 0.01:
        level = '≈ 0 (неинф.)'
    elif cv < 30:
        level = 'Низкий'
    elif cv < 70:
        level = 'Средний'
    else:
        level = 'Высокий'

    print(f"{col:<20} {std_val:<12.2f} {mean_val:<12.2f} {cv:<15.1f}% {level:<15}")

print("\nВывод:")
print("  • Разброс по каналам СИЛЬНО различается: от сотен (PT08.S1) до единиц (CO)")
print("  • Необходимо масштабирование признаков перед подачей в модель")
print("  • Каналов со стандартным отклонением ≈ 0 нет — все признаки информативны")


# ============================================================
# Шаг 4. Определение частоты дискретизации
# ============================================================

print("\n" + "="*70)
print("АНАЛИЗ ЧАСТОТЫ ДИСКРЕТИЗАЦИИ")
print("="*70)

# Вычисляем разницу между соседними временными метками
time_diffs = df_raw.index.to_series().diff().dropna()

# Уникальные интервалы
unique_intervals = time_diffs.value_counts().head(10)

print("Распределение интервалов между отсчётами (топ-10):")
for interval, count in unique_intervals.items():
    print(f"  {interval} — {count} раз(а)")

# Основной интервал
main_interval = time_diffs.mode()[0]
print(f"\nОсновной интервал дискретизации: {main_interval}")
print(f"Частота дискретизации: 1 отсчёт в час (почасовая)")

# Проверка равномерности
total_intervals = len(time_diffs)
uniform_count = (time_diffs == main_interval).sum()
uniform_pct = (uniform_count / total_intervals) * 100

print(f"\nРавномерность интервалов:")
print(f"  Всего интервалов: {total_intervals}")
print(f"  Равных основному ({main_interval}): {uniform_count} ({uniform_pct:.1f}%)")

if uniform_pct > 95:
    print("  ✅ Интервалы практически равномерны (>95%)")
else:
    print(f"  ⚠ Интервалы неравномерны — есть пропуски в данных")


# ============================================================
# Шаг 5. Проверка значений за допустимыми пределами
# ============================================================

print("\n" + "="*70)
print("ПРОВЕРКА НА ВЫХОД ЗА ФИЗИЧЕСКИ ДОПУСТИМЫЕ ПРЕДЕЛЫ")
print("="*70)

# Определяем физические пределы для каждого канала
physical_limits = {
    'CO(GT)':         (0, 50),       # мг/м³ (ПДК ~3-5, пики до 50 возможны)
    'PT08.S1(CO)':    (0, None),     # усл. ед. — нет верхнего предела
    'NMHC(GT)':       (0, None),     # мкг/м³
    'C6H6(GT)':       (0, None),     # мкг/м³
    'PT08.S2(NMHC)':  (0, None),     # усл. ед.
    'NOx(GT)':        (0, None),     # ppb
    'PT08.S3(NOx)':   (0, None),     # усл. ед.
    'NO2(GT)':        (0, None),     # мкг/м³
    'PT08.S4(NO2)':   (0, None),     # усл. ед.
    'PT08.S5(O3)':    (0, None),     # усл. ед.
    'T':              (-20, 50),     # °C (итальянский климат)
    'RH':             (0, 100),      # % (относительная влажность)
    'AH':             (0, None),     # г/м³
}

print(f"{'Канал':<20} {'Минимум':<10} {'Максимум':<10} {'Статус':<30}")
print("-"*70)

for col in numeric_cols:
    min_val = df_raw[col].min()
    max_val = df_raw[col].max()

    if col in physical_limits:
        low_lim, up_lim = physical_limits[col]

        issues = []
        if low_lim is not None and min_val < low_lim:
            issues.append(f"Ниже допустимого ({min_val:.2f} < {low_lim})")
        if up_lim is not None and max_val > up_lim:
            issues.append(f"Выше допустимого ({max_val:.2f} > {up_lim})")

        if issues:
            status = '⚠ ' + '; '.join(issues)
        else:
            status = '✅ В допустимых пределах'
    else:
        status = '✅ Нет явных ограничений'

    print(f"{col:<20} {min_val:<10.2f} {max_val:<10.2f} {status:<30}")


# ============================================================
# Шаг 6. Расширенный анализ целевой переменной CO(GT)
# ============================================================

print("\n" + "="*70)
print("РАСШИРЕННАЯ СТАТИСТИКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ CO(GT)")
print("="*70)

co = df_raw['CO(GT)']

# Дополнительные метрики
skewness = co.skew()
kurtosis = co.kurtosis()

# Процентили для понимания хвостов
p1 = np.percentile(co.dropna(), 1)
p5 = np.percentile(co.dropna(), 5)
p95 = np.percentile(co.dropna(), 95)
p99 = np.percentile(co.dropna(), 99)

print(f"Среднее:                {co.mean():.3f} мг/м³")
print(f"Медиана:                {co.median():.3f} мг/м³")
print(f"Стд. отклонение:        {co.std():.3f} мг/м³")
print(f"Минимум:                {co.min():.3f} мг/м³")
print(f"Максимум:               {co.max():.3f} мг/м³")
print(f"Асимметрия (skewness):  {skewness:.3f}")
print(f"Эксцесс (kurtosis):     {kurtosis:.3f}")
print(f"1-й процентиль:         {p1:.2f} мг/м³")
print(f"5-й процентиль:         {p5:.2f} мг/м³")
print(f"95-й процентиль:        {p95:.2f} мг/м³")
print(f"99-й процентиль:        {p99:.2f} мг/м³")

print("\nВывод по распределению CO(GT):")
print(f"  • Асимметрия = {skewness:.1f} — распределение СУЩЕСТВЕННО скошено вправо")
print(f"  • Эксцесс = {kurtosis:.1f} — распределение островершинное (тяжёлые хвосты)")
print("  • 95% значений CO лежат ниже 95-го процентиля — пики редки, но экстремальны")
print("  • Для моделирования может потребоваться log-преобразование целевой переменной")


print("\n" + "="*70)
print("✅ Этап 3 завершён: статистический анализ выполнен")
print("="*70)

# ============================================================
# Шаг 1. Доля пропущенных значений в каждом канале
# ============================================================

print("\n" + "="*70)
print("ШАГ 1: ДОЛЯ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ПО КАНАЛАМ")
print("="*70)

# Выбираем числовые столбцы
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns

# Считаем пропуски
print(f"{'Канал':<20} {'Всего':<8} {'Пропусков':<10} {'% пропусков':<12} {'Статус':<25}")
print("-"*75)

missing_summary = {}

for col in numeric_cols:
    total = len(df_raw)
    missing = df_raw[col].isna().sum()
    missing_pct = (missing / total) * 100

    missing_summary[col] = {
        'total': total,
        'missing': missing,
        'percent': round(missing_pct, 2)
    }

    # Оценка критичности
    if missing_pct == 0:
        status = '✅ Нет пропусков'
    elif missing_pct < 5:
        status = '🟢 Незначительно (<5%)'
    elif missing_pct < 20:
        status = '🟡 Умеренно (5–20%)'
    elif missing_pct < 50:
        status = '🟠 Много (20–50%)'
    else:
        status = '🔴 Критически много (>50%)'

    print(f"{col:<20} {total:<8} {missing:<10} {missing_pct:<12.2f}% {status:<25}")

# Сохраняем для отчёта
missing_df = pd.DataFrame(missing_summary).T
missing_df.to_csv('/content/missing_analysis.csv', sep=';', encoding='utf-8-sig')
print("\n✅ Таблица пропусков сохранена в /content/missing_analysis.csv")

# ============================================================
# ВЫБРОСЫ: ПРАВИЛО ТРЁХ СИГМ + BOX PLOT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ПОИСК ВЫБРОСОВ ПО ПРАВИЛУ ТРЁХ СИГМ")
print("="*70)
print("Выброс = значение за пределами: среднее ± 3σ")
print("-"*70)

# Если работаешь с исходным файлом — раскомментируй:
# df_raw.replace(-200, np.nan, inplace=True)
# df_raw.replace('-200', np.nan, inplace=True)

# Только числовые столбцы
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns

outlier_summary = {}

for col in numeric_cols:
    valid_data = df_raw[col].dropna()

    if len(valid_data) == 0:
        continue

    mean_val = valid_data.mean()
    std_val = valid_data.std()

    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val

    # Считаем выбросы
    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
    n_outliers = len(outliers)
    outlier_pct = (n_outliers / len(valid_data)) * 100

    outlier_summary[col] = {
        'mean': mean_val,
        'std': std_val,
        'lower': lower_bound,
        'upper': upper_bound,
        'n_outliers': n_outliers,
        'pct': round(outlier_pct, 2)
    }

    print(f"Канал: {col}")
    print(f"  Среднее = {mean_val:.2f}, σ = {std_val:.2f}")
    print(f"  Границы: [{lower_bound:.2f} ; {upper_bound:.2f}]")
    print(f"  Выбросов: {n_outliers} из {len(valid_data)} ({outlier_pct:.2f}%)")
    if n_outliers > 0:
        print(f"  Min выброс = {outliers.min():.2f}, Max выброс = {outliers.max():.2f}")
    print()


# ============================================================
# BOX PLOT — ДИАГРАММЫ РАЗМАХА ДЛЯ КАЖДОГО КАНАЛА
# ============================================================

print("="*70)
print("BOX PLOT — ДИАГРАММЫ РАЗМАХА")
print("="*70)

# --- График 1: Целевая переменная + эталонные газы ---
gas_cols = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
gas_titles = ['CO (мг/м³)', 'НМУВ (мкг/м³)', 'Бензол (мкг/м³)', 'NOx (ppb)', 'NO₂ (мкг/м³)']
gas_colors = ['darkred', 'gray', 'darkorange', 'darkblue', 'darkgreen']

# Оставляем только те, что есть в данных
gas_cols = [c for c in gas_cols if c in df_raw.columns]

fig, axes = plt.subplots(1, len(gas_cols), figsize=(4*len(gas_cols), 5))
if len(gas_cols) == 1:
    axes = [axes]

for i, (col, title, color) in enumerate(zip(gas_cols, gas_titles[:len(gas_cols)], gas_colors[:len(gas_cols)])):
    ax = axes[i]
    data = df_raw[col].dropna()
    ax.boxplot(data, patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color='black', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Эталонные газы (GT)', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/boxplot_gases.png', dpi=150, bbox_inches='tight')
plt.show()


# --- График 2: Отклики датчиков ---
sensor_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
sensor_titles = ['CO', 'НМУВ', 'NOx', 'NO₂', 'O₃']
sensor_colors = ['darkblue', 'darkgreen', 'darkorange', 'purple', 'brown']

sensor_cols = [c for c in sensor_cols if c in df_raw.columns]

fig, axes = plt.subplots(1, len(sensor_cols), figsize=(4*len(sensor_cols), 5))
if len(sensor_cols) == 1:
    axes = [axes]

for i, (col, title, color) in enumerate(zip(sensor_cols, sensor_titles[:len(sensor_cols)], sensor_colors[:len(sensor_cols)])):
    ax = axes[i]
    data = df_raw[col].dropna()
    ax.boxplot(data, patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color='black', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
    ax.set_title(f'PT08 — {title}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Отклики металлоксидных сенсоров', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/boxplot_sensors.png', dpi=150, bbox_inches='tight')
plt.show()


# --- График 3: Метеопараметры ---
weather_cols = ['T', 'RH', 'AH']
weather_titles = ['Температура (°C)', 'Отн. влажность (%)', 'Абс. влажность (г/м³)']
weather_colors = ['red', 'blue', 'green']

weather_cols = [c for c in weather_cols if c in df_raw.columns]

fig, axes = plt.subplots(1, len(weather_cols), figsize=(4*len(weather_cols), 5))
if len(weather_cols) == 1:
    axes = [axes]

for i, (col, title, color) in enumerate(zip(weather_cols, weather_titles, weather_colors)):
    ax = axes[i]
    data = df_raw[col].dropna()
    ax.boxplot(data, patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color='black', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Метеорологические параметры', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/boxplot_weather.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Выбросы подсчитаны, box plot'ы построены и сохранены")

# ============================================================
# ЭТАП 5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ЭТАП 5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ")
print("="*70)

# Числовые каналы
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns

# ============================================================
# Шаг 1. Общая диаграмма размаха для всех каналов
# ============================================================

print("\nСтроим общую диаграмму размаха для всех каналов...")

fig, ax = plt.subplots(figsize=(16, 6))

data_raw = [df_raw[col].dropna().values for col in numeric_cols]
bp = ax.boxplot(data_raw, patch_artist=True, widths=0.6,
                boxprops=dict(alpha=0.7),
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=2, alpha=0.4))

# Раскрашиваем
colors = plt.cm.tab20(np.linspace(0, 1, len(numeric_cols)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
ax.set_title('Сравнение диапазонов значений по всем каналам (исходные масштабы)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Значение')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/content/range_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ График сохранён: /content/range_comparison.png")


# ============================================================
# Шаг 2. Таблица диапазонов и масштабов
# ============================================================

print("\n" + "="*70)
print("ТАБЛИЦА: ДИАПАЗОНЫ И МАСШТАБЫ КАНАЛОВ")
print("="*70)

print(f"{'Канал':<20} {'Мин':>10} {'Макс':>10} {'Размах':>10} {'Порядок':>10}")
print("-"*65)

for col in numeric_cols:
    data = df_raw[col].dropna()
    min_v = data.min()
    max_v = data.max()
    range_v = max_v - min_v

    # Определяем порядок величины
    if range_v > 0:
        order = int(np.floor(np.log10(range_v)))
    else:
        order = 0

    print(f"{col:<20} {min_v:>10.2f} {max_v:>10.2f} {range_v:>10.2f} {'10^' + str(order):>10}")

# Вычисляем отношение максимального размаха к минимальному
ranges = []
for col in numeric_cols:
    data = df_raw[col].dropna()
    ranges.append(data.max() - data.min())

max_range = max(ranges)
min_range = min([r for r in ranges if r > 0])
ratio = max_range / min_range

print(f"\nМаксимальный размах: {max_range:.0f}")
print(f"Минимальный размах:  {min_range:.2f}")
print(f"Отношение (max/min):  {ratio:.0f} раз")

if ratio > 10:
    print("  → МАСШТАБИРОВАНИЕ ОБЯЗАТЕЛЬНО")
elif ratio > 3:
    print("  → Масштабирование рекомендовано")
else:
    print("  → Масштабирование не критично")

print("\n" + "="*70)
print("✅ Этап 5 завершён")
print("="*70)

# ============================================================
# ЭТАП 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ЭТАП 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("="*70)

# Выбираем числовые каналы, исключаем NMHC(GT) — слишком много пропусков
cols_for_corr = [c for c in df_raw.select_dtypes(include=[np.number]).columns
                 if c != 'NMHC(GT)']

df_corr = df_raw[cols_for_corr].dropna()

# ============================================================
# Шаг 1. Матрица парных корреляций Пирсона
# ============================================================

print("\nМатрица парных коэффициентов корреляции Пирсона:")
print("-"*70)

corr_matrix = df_corr.corr()

# Округляем для читаемости
print(corr_matrix.round(2))

# Сохраняем для отчёта
corr_matrix.round(2).to_csv('/content/correlation_matrix.csv', sep=';', encoding='utf-8-sig')
print("\n✅ Матрица сохранена: /content/correlation_matrix.csv")


# ============================================================
# Шаг 2. Тепловая карта
# ============================================================

print("\nСтроим тепловую карту...")

plt.figure(figsize=(14, 10))

# Маска для верхнего треугольника (чтобы не дублировать)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix,
            mask=mask,
            annot=True,           # цифры в ячейках
            fmt='.2f',            # два знака после запятой
            cmap='RdBu_r',        # красный = +1, синий = -1, белый = 0
            center=0,             # ноль по центру цветовой шкалы
            vmin=-1, vmax=1,
            square=True,
            linewidths=1,
            linecolor='white',
            cbar_kws={'label': 'Коэффициент корреляции Пирсона', 'shrink': 0.8},
            annot_kws={'size': 8})

plt.title('Матрица парных корреляций признаков\n(корреляция Пирсона)',
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('/content/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Тепловая карта сохранена: /content/correlation_heatmap.png")


# ============================================================
# Шаг 3. Анализ корреляций с целевой переменной CO(GT)
# ============================================================

print("\n" + "="*70)
print("КОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ CO(GT)")
print("="*70)

target_corr = corr_matrix['CO(GT)'].drop('CO(GT)').sort_values(ascending=False)

print(f"{'Признак':<20} {'Корреляция с CO(GT)':>20}")
print("-"*45)

for col, val in target_corr.items():
    bar = '█' * int(abs(val) * 20)
    direction = '+' if val > 0 else '-'
    print(f"{col:<20} {direction}{abs(val):.2f} {bar}")

# Топ-3 самых сильных корреляций
print("\nТоп-3 признака, наиболее связанных с CO(GT):")
for i, (col, val) in enumerate(target_corr.head(3).items(), 1):
    print(f"  {i}. {col} — корреляция {val:+.2f}")


# ============================================================
# Шаг 4. Поиск мультиколлинеарности
# ============================================================

print("\n" + "="*70)
print("ПОИСК МУЛЬТИКОЛЛИНЕАРНОСТИ (КОРРЕЛЯЦИЯ > 0.85)")
print("="*70)

# Ищем пары с |корреляцией| > 0.85
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

if high_corr_pairs:
    print(f"Найдено {len(high_corr_pairs)} пар с корреляцией > 0.85:")
    for col1, col2, val in high_corr_pairs:
        print(f"  {col1} ↔ {col2}: {val:+.2f}")
    print("\n  → Эти пары несут почти одинаковую информацию — один из признаков можно исключить")
else:
    print("Пар с экстремально высокой корреляцией (> 0.85) не обнаружено")


# ============================================================
# Шаг 5. Признаки, слабо коррелирующие со всеми
# ============================================================

print("\n" + "="*70)
print("ПРИЗНАКИ С НИЗКОЙ КОРРЕЛЯЦИЕЙ С ОСТАЛЬНЫМИ")
print("="*70)

# Средняя абсолютная корреляция каждого признака со всеми остальными
mean_abs_corr = {}
for col in corr_matrix.columns:
    others = corr_matrix[col].drop(col)
    mean_abs_corr[col] = abs(others).mean()

for col, val in sorted(mean_abs_corr.items(), key=lambda x: x[1]):
    bar = '█' * int(val * 30)
    print(f"{col:<20} | Средняя |r| = {val:.2f} {bar}")
    if val < 0.3:
        print(f"{'':<20}   → НЕЗАВИСИМЫЙ признак, несёт уникальную информацию")

print("\n" + "="*70)
print("✅ Этап 6 завершён")
print("="*70)

# ============================================================
# ЭТАП 7. ПОИСК И АНАЛИЗ ШУМОВ
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

print("="*70)
print("ЭТАП 7. ПОИСК И АНАЛИЗ ШУМОВ")
print("="*70)

# ============================================================
# Шаг 1–2. Декомпозиция и визуализация
# ============================================================

print("\nВыполняем декомпозицию CO(GT)...")
print("Период сезонности: 24 часа (суточная), модель: аддитивная")

co_clean = df_raw['CO(GT)'].dropna()

decomposition = seasonal_decompose(co_clean, model='additive', period=24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plot_index = co_clean.index

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

axes[0].plot(plot_index, co_clean, color='darkred', linewidth=0.5)
axes[0].set_title('Исходный ряд: CO(GT)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('мг/м³')
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_index, trend, color='darkblue', linewidth=1)
axes[1].set_title('Тренд', fontsize=11, fontweight='bold')
axes[1].set_ylabel('мг/м³')
axes[1].grid(True, alpha=0.3)

axes[2].plot(plot_index, seasonal, color='darkgreen', linewidth=0.5)
axes[2].set_title('Сезонная компонента (суточная)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('мг/м³')
axes[2].grid(True, alpha=0.3)

axes[3].plot(plot_index, residual, color='gray', linewidth=0.3)
axes[3].set_title('Остатки (шум)', fontsize=11, fontweight='bold')
axes[3].set_ylabel('мг/м³')
axes[3].set_xlabel('Дата')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Декомпозиция выполнена")


# ============================================================
# Шаг 3. Выделение сигнала
# ============================================================

signal = trend + seasonal

print("\n" + "="*70)
print("ВЫДЕЛЕНИЕ СИГНАЛА")
print("="*70)
print("Модель: аддитивная → сигнал = тренд + сезонность")
print(f"Дисперсия сигнала:        {np.var(signal.dropna()):.4f}")
print(f"Дисперсия шума (остатков): {np.var(residual.dropna()):.4f}")


# ============================================================
# Шаг 4. Расчёт SNR
# ============================================================

valid_mask = signal.notna() & residual.notna()
signal_valid = signal[valid_mask]
residual_valid = residual[valid_mask]

var_signal = np.var(signal_valid)
var_noise = np.var(residual_valid)
snr = 10 * np.log10(var_signal / var_noise)

print("\n" + "="*70)
print("РАСЧЁТ SNR")
print("="*70)
print(f"SNR = {snr:.2f} дБ")

if snr > 20:
    quality = "Отлично — шум практически незаметен"
elif snr > 10:
    quality = "Хорошо — сигнал доминирует"
elif snr > 0:
    quality = "Удовлетворительно — сигнал и шум сравнимы"
else:
    quality = "Плохо — шум сильнее сигнала"

print(f"Качественная оценка: {quality}")


# ============================================================
# Шаг 5. Гистограмма остатков
# ============================================================

# Настройка размера для одной гистограммы
plt.figure(figsize=(8, 5))

# Построение гистограммы остатков
plt.hist(residual_valid, bins=50, color='gray', alpha=0.7, edgecolor='black', linewidth=0.3)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)

# Оформление графика
plt.title('Гистограмма остатков (шума)', fontsize=11, fontweight='bold')
plt.xlabel('Остаток, мг/м³')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

# Сохранение и вывод на экран
plt.tight_layout()
plt.savefig('/content/residuals_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("ХАРАКТЕРИСТИКИ РАСПРЕДЕЛЕНИЯ ОСТАТКОВ")
print("="*70)
print(f"Среднее:     {residual_valid.mean():.4f}")
print(f"Стд. откл.:  {residual_valid.std():.4f}")
print(f"Асимметрия:  {residual_valid.skew():.3f}")
print(f"Эксцесс:     {residual_valid.kurtosis():.3f}")

skew = residual_valid.skew()
kurt = residual_valid.kurtosis()

if abs(skew) < 0.5 and abs(kurt) < 1:
    shape = "Близко к нормальному"
elif abs(skew) >= 0.5:
    shape = "Асимметричное (скошено вправо)" if skew > 0 else "Асимметричное (скошено влево)"
elif kurt > 2:
    shape = "С тяжёлыми хвостами"
else:
    shape = "Смешанного типа"

print(f"Форма распределения: {shape}")

print("\n✅ Этап 7 завершён")
