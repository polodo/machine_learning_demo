from binance.client import Client
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import csv
from mysite.crawl_binance_data.utils import printProgressBar
import datetime

urllib3.disable_warnings()
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

SECRET_KEY = "oOdLWuoRJVWYpEMtj6arcC1iCOK9nHd0lg5mIcKrOYXEBLpBOh9ZaRYSfwAR5Eht"
API_KEY = "7bT35R9Lq0ZnV1AqT95mx5jO6CI1U1iAzCNtrsG0tOka8VwIMGUbV5iAsbVD32dR"

SYMBOL = "BTCUSDT"  # string, exchange symbol.
INTERVAL_STR = Client.KLINE_INTERVAL_30MINUTE # string, param is put in request to send to Binance to get records.
INTERVAL = 30 # int, minute to get data for series
MAX_ROW = 500
MAX_RECORDS = MAX_ROW * 24 * 2 # get data in 500 days

client = Client(API_KEY, SECRET_KEY, {"verify": False, "timeout": 20})

# print(client.__dict__)
flag = True
# test to get first data
candles = client.get_klines(symbol=SYMBOL,
                            interval=INTERVAL_STR,
                            limit=MAX_ROW)

# exit()
data = []
time_series = []
if not candles:
    exit(1)
else:
    data = [float(line[4]) for line in candles]
    time_series = [int(line[0]) for line in candles]
    # print(candles[0][0], candles[0][6], int(candles[0][6]) - int(candles[0][0]))
    next_end_time = int(candles[0][0] - 1)
    next_start_time = next_end_time - INTERVAL * 60 * 1000 * MAX_ROW
    # print(next_end_time, next_start_time, next_end_time - next_start_time)
printProgressBar(0, 48, prefix='Progress:', suffix='Complete', length=48)
while flag:
    # print("download other packs. %s - %s" % (datetime.datetime.fromtimestamp(int(next_start_time / 1000)),
    #                                          datetime.datetime.fromtimestamp(int(next_end_time / 1000))))
    candles = client.get_klines(symbol='BTCUSDT',
                                interval=Client.KLINE_INTERVAL_30MINUTE,
                                limit=MAX_ROW,
                                startTime=next_start_time,
                                endTime=next_end_time)
    # print("Get %d records from BINANCE" % (len(candles)))
    # print(len(data), len(candles))
    if not candles or len(data) >= MAX_RECORDS:
        flag = False
    else:
        data_temp = [float(line[4]) for line in candles]
        time_series_temp = [int(line[0]) for line in candles]
        data = data_temp + data
        time_series = time_series_temp + time_series
        next_end_time = int(candles[0][0] - 1)
        next_start_time = next_end_time - INTERVAL * 60 * 1000 * MAX_ROW + 1
        printProgressBar(int(len(data) / MAX_ROW), 48, prefix = 'Download DATA:', suffix = 'Complete', length = 48)

with open("BTCUSDT_30mins.csv", "w+", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in data:
        spamwriter.writerow([line])

with open("BTCUSDT_30mins_time.csv", "w+", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in time_series:
        spamwriter.writerow([line])