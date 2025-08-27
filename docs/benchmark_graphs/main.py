import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def create_duration_graph(data, target, title):
    dates = []
    durations = []

    for day, duration in data:
        dates.append(datetime.strptime(day, '%Y-%m-%d'))
        durations.append(duration)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, durations, marker='o', linewidth=2,
            markersize=7, color='#2E86AB')

    min_date = min(dates)
    max_date = max(dates)
    date_range = max_date - min_date
    if date_range < timedelta(days=30):
        max_date = min_date + timedelta(days=30)
    ax.set_xlim(min_date - timedelta(days=1), max_date + timedelta(days=1))

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.axhline(y=target, color='green', linestyle='--',
               linewidth=2, label=f'Target ({target} s)')

    ax.set_ylabel('Proving time (s)', fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_ylim(0, max(durations) * 1.1)

    plt.tight_layout()
    file_name = title.replace(" ", "_").lower()
    plt.savefig(f'graphs/{file_name}.svg', format='svg', bbox_inches='tight')


if __name__ == "__main__":

    create_duration_graph(data=[
        ('2025-08-27', 2.7),
    ], target=0.25, title="Recursive WHIR opening")

    create_duration_graph(data=[
        ('2025-08-27', 14),
    ], target=0.5, title="500 XMSS aggregated")
