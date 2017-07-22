import numpy as np
import urllib.request
from bs4 import BeautifulSoup


def retrieve_monthly_clearness_block(block, month):
    ([latmax, lonmin], [latmin, lonmax]) = block

    url = "https://eosweb.larc.nasa.gov/cgi-bin/sse/subset.cgi?email=skip%40larc.nasa.gov&latmin=" \
          + str(latmin) + "&lonmin=" + str(lonmin) + "&latmax=" + \
          str(latmax) + "&lonmax=" + str(lonmax) + \
          "&month=" + str(month) + "&tenyear=avg_kt&grid=none&submit=Submit"
    with urllib.request.urlopen(url) as response:
        html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        cells = [cell.text.strip() for cell in soup("td")]

        if lonmax == 180:
            blockdata = [float(k) for k in cells[100:100 + 91 * 89 + 91]]
            npblock = np.delete(np.array(blockdata).reshape(90, 91), (0), axis=1)
        else:
            blockdata = [float(k) for k in cells[101:101 + 91 * 90 + 90]]
            npblock = np.delete(np.array(blockdata).reshape(90, 92), (0, 91, 92), axis=1)

    return npblock


def retrieve_monthly_clearness_index(month):
    allpoints = []
    for lat in [90, 0, -90]:
        for lon in [-180, -90, 0, 90, 180]:
            allpoints.append([lat, lon])
    blocks = []
    for i, j in zip(allpoints[0:9], allpoints[6:16]):
        blocks.append([i, j])
    del blocks[4]

    upperblock = np.concatenate(
        (retrieve_monthly_clearness_block(blocks[0], month),
         retrieve_monthly_clearness_block(blocks[1], month),
         retrieve_monthly_clearness_block(blocks[2], month),
         retrieve_monthly_clearness_block(blocks[3], month),
         ), axis=1)

    lowerblock = np.concatenate(
        (retrieve_monthly_clearness_block(blocks[4], month),
         retrieve_monthly_clearness_block(blocks[5], month),
         retrieve_monthly_clearness_block(blocks[6], month),
         retrieve_monthly_clearness_block(blocks[7], month),
         ), axis=1)

    globalblock = np.concatenate((upperblock, lowerblock), axis=0)

    return globalblock

globalavgkt = np.dstack([retrieve_monthly_clearness_index(month+1)\
                for month in range(12)])

np.save('globalavgkt1.npy', globalavgkt)