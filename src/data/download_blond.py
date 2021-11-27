import os
from ftplib import FTP

from tqdm import tqdm

from src.__init__ import ROOT_DIR


def find(a_list, keyword):
    """ Return an item of a list containing a keyword

    Args:
        a_list (list): 1-5
        keyword (string): train, val, test, all
    Returns:
        item (string): item containing keyword
    """
    for item in a_list:
        if item in keyword or keyword in item:
            return item
    return None


def download_summary(ip, user, passwd, socket_base):
    """ Downloads all summery data for 50khz or 250khz data

    Args:
        ip (string): IP of the FTP
        user (string): User of the FTP
        passwd (string): Password of the FTP
        socket_base (string): /BLOND/BLOND-50 or  /BLOND/BLOND-250
    Returns:
    """
    base_dir = os.path.join(ROOT_DIR, 'data/BLOND')
    os.makedirs(base_dir, exist_ok=True)

    ftp_socket = FTP(ip)
    ftp_socket.login(user, passwd)

    file_name = 'BLOND/appliance_log.json'
    file_path = os.path.join(base_dir, 'appliance_log.json')
    file = open(file_path, 'wb')
    ftp_socket.retrbinary('RETR %s' % file_name, file.write)

    ftp_socket.cwd(socket_base)
    dates = ftp_socket.nlst()
    for date in tqdm(dates):
        ftp_socket.cwd(date + '/clear')
        files = ftp_socket.nlst()
        file_name = find(files, 'summary')

        if file_name is not None:
            directory = os.path.join(ROOT_DIR, 'data', socket_base, date, 'clear')
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, file_name)
            if not os.path.exists(file_path):
                file = open(file_path, 'wb')
                ftp_socket.retrbinary('RETR %s' % file_name, file.write)

        ftp_socket.cwd('..')

        for medal_id in range(1, 16):
            medal = 'medal-{}/'.format(medal_id)
            ftp_socket.cwd(medal)
            files = ftp_socket.nlst()
            file_name = find(files, 'summary')

            if file_name is not None:
                directory = os.path.join(ROOT_DIR, 'data', socket_base, date, medal)
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, file_name)
                if not os.path.exists(file_path):
                    print('downloading')
                    file = open(file_path, 'wb')
                    ftp_socket.retrbinary('RETR %s' % file_name, file.write)

            ftp_socket.cwd('..')
        ftp_socket.cwd('..')


def download_day(ip, user, passwd, socket_base, date, medal_id):
    """ Downloads all high frequency data for one day of all six sockets

    Args:
        ip (string): IP of the FTP
        user (string): User of the FTP
        passwd (string): Password of the FTP
        socket_base (string): /BLOND/BLOND-50 or  /BLOND/BLOND-250
        date (string): String representation of the date
        medal_id (int): Medal index
    Returns:
    """
    base_dir = os.path.join(ROOT_DIR, 'data/BLOND')
    os.makedirs(base_dir, exist_ok=True)

    ftp_socket = FTP(ip)
    ftp_socket.login(user, passwd)

    medal = 'medal-{}/'.format(medal_id)
    medal_path = os.path.join(socket_base, date + '/', medal)
    ftp_socket.cwd(medal_path)

    directory = os.path.join(ROOT_DIR, 'data', socket_base, date, medal)
    os.makedirs(directory, exist_ok=True)

    files = ftp_socket.nlst()
    for file_name in tqdm(files):
        if '.pdf' not in file_name:
            file_path = os.path.join(directory, file_name)
            file = open(file_path, 'wb')
            ftp_socket.retrbinary('RETR %s' % file_name, file.write)


if __name__ == "__main__":
    socket_base = 'BLOND/BLOND-50/'
    ip = '138.246.224.34'
    user = 'm1375836'
    passwd = 'm1375836'

    download_summary(ip, user, passwd, socket_base)
