import csv, calendar, datetime

DATA_FILENAME = 'data/weather.csv'
PARSED_FILENAME = 'data/machineweather.csv'

in_header = []
    
out_header = [
    'Day thru Year',
    'Temperature',
    'Relative Humidity'
    ]
    #TODO


def divide_date(year, month, day) -> float:
    """Returns the date as a float indicating how much of the year
    is finished on that date."""
    month_days = [
        31, 
        29 if calendar.isleap(year) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
        ]
    days_elapsed = sum(month_days[:round(month-1)]) + day
    days_tot = sum(month_days)
    
    return days_elapsed / days_tot
    
def conv_csvdata_to_dict(headers: list, data: list) -> dict:
    """Converts one row of CSV data to a dict for easy subscripting."""
    ret = {}
    for header, datum in zip(headers, data):
        try:
            ret[header] = float(datum)
        except ValueError:
            ret[header] = datum
    return ret

with open(DATA_FILENAME, newline='') as in_file:
    with open(PARSED_FILENAME, 'w', newline='') as out_file:
        in_csv = csv.reader(in_file)
        out_csv = csv.writer(out_file, quoting=csv.QUOTE_NONNUMERIC)
        
        #write header
        out_csv.writerow(out_header)
        
        #consume header of input
        in_header = next(in_csv)
        
        for in_datum_raw in in_csv:
            in_datum = conv_csvdata_to_dict(in_header, in_datum_raw)
            out_csv.writerow(
                [divide_date(in_datum['Year'], in_datum['Month'], in_datum['Day']),
                 in_datum['Temp (Â°C)'],
                 in_datum['Rel Hum (%)']
                ]
                )
                