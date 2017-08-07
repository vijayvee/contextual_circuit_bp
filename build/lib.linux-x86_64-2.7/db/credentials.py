
def postgresql_credentials():
    return {
            'username': 'contextual_DCN',
            'password': 'serrelab'
           }


def postgresql_connection(port=''):
    unpw = postgresql_credentials()
    params = {
        'database': 'contextual_DCN',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params


def machine_credentials():
    return {
        'username': 'drew',
        'password': 'serrelab',
       }


def clusteer_credentials():
    return {
        'username': 'drew',
        'password': 'serrelab',
        'ssh_address': 'serrep3.services.brown.edu'
       }

