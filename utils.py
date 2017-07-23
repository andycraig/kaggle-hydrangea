import datetime

filesafe_replacements = str.maketrans(" :", "_-")

def datetime_for_filename():
    return str(datetime.datetime.now()).translate(filesafe_replacements)
