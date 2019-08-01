import requests as req
import hmac
import hashlib
import json
import os
import time;

def hash_rq_content_to_signature(secret_key, data):
    """
    :param secret_key: secret key is combined to signature
    :param data: dictionary of data list in signature, it should be extracted from url
    :return: url with signature
    """
    msg = ""
    lst = []
    for item in iter(data.items()):
        sub_msg = "{}={}".format(item[0], item[1])
        lst.append(sub_msg)
    msg = "&".join(lst)
    signature = hmac.new(bytes(secret_key, 'utf-8'),
                         msg=bytes(msg, 'utf-8'),
                         digestmod=hashlib.sha256).hexdigest()
    return msg + "&signature=" + signature

ts = int(time.time())
SECRET_KEY = "oOdLWuoRJVWYpEMtj6arcC1iCOK9nHd0lg5mIcKrOYXEBLpBOh9ZaRYSfwAR5Eht"
API_KEY = "7bT35R9Lq0ZnV1AqT95mx5jO6CI1U1iAzCNtrsG0tOka8VwIMGUbV5iAsbVD32dR"
test_key = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
HASHED_SECRET_KEY = hmac.new(bytes(test_key, 'utf-8'),
                             msg=bytes("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559", 'utf-8'),
                             digestmod=hashlib.sha256).hexdigest()
print (HASHED_SECRET_KEY)
base_api = "https://api.binance.com"
order_api = "/api/v3/order?symbol={couple_coins}&side={interaction}&type={type}&timeInForce=GTC&quantity={qty}&price={price_lmt}&recvWindow={recv_time}&timestamp={timestamp}&signature={hashed_key}"
get_time_api = "/api/v1/time"
base_header = {#'X-MBX-USED-WEIGHT': '2',
               'X-MBX-APIKEY': API_KEY}
'''
Security Type: Description
NONE: Endpoint can be accessed freely.
TRADE: Endpoint requires sending a valid API-Key and signature.
USER_DATA: Endpoint requires sending a valid API-Key and signature.
USER_STREAM: Endpoint requires sending a valid API-Key.
MARKET_DATA: Endpoint requires sending a valid API-Key.
'''

security_type = ['NONE', 'TRADE', 'USER_DATA', 'USER_STREAM', 'MARKET_DATA']
main_couple_coins = ['ETHUSDT', 'ADAUSDT']
interaction = 'SELL'

def get_server_time():
    try:
        r = req.get(base_api + get_time_api)
        return r.json().get('serverTime')
    except Exception as e :
        return {"msg": e,
                "code": 501}

timestamp = get_server_time()
request_order = {'symbol': main_couple_coins[1],
                 'recvWindow': 3000,
                 'timestamp': timestamp,
                 'symbol': main_couple_coins[1]}
signature_url = hash_rq_content_to_signature(SECRET_KEY, request_order)
print(signature_url)
# opened_orders = "/api/v3/openOrders?symbol={couple_coins}&recvWindow={recv_time}&timestamp={ts}&signature={hashed_key}".format(couple_coins=main_couple_coins[1],
#                                         recv_time=3000,
#                                         ts=timestamp,
#                                         hashed_key=HASHED_SECRET_KEY)

r = req.get(base_api + "/api/v3/openOrders?" + signature_url, headers = base_header)

get_data = '/api/v1/klines?symbol=ADAUSDT&interval=500&signature=8d696fe10ccaa7a06438090de1b096bdbd80e89f3e16f5c60380a08de928ee25'


r = req.get(base_api + get_data , headers = base_header)

print(r.__dict__)
