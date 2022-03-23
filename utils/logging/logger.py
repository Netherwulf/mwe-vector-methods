import datetime


def log_message(*message):
    date = datetime.date.today().strftime("%d/%m/%Y")

    time = datetime.datetime.now().strftime("%H:%M:%S")

    message = '\n'.join(message)

    print(f'{date} - {time} : {message}')
