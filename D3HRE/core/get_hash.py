import hashlib
# MD5 hash checksum is use to link a mission and download file
# The idea is it is hard to give a name of a route or just by name them with
# first way points. We can check the sum of a mission, it will be a unique
# number that help us to track the download file for each mission

def hash_value(data):
    """
    Get hash of python object

    :param data: object or data
    :return: the hash value for the data (full length)
    """
    hashId = hashlib.md5()
    hashId.update(repr(data).encode('utf-8'))
    return hashId.hexdigest()

def hash_raw(data):
    hashId = hashlib.md5()
    hashId.update(repr(data).encode('utf-8'))
    return hashId