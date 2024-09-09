import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dblp = pd.read_json('pre/dblp_d_spsbert.json', lines=True, orient='records')
pmc = pd.read_json('pre/pmc_d_spsbert.json', lines=True, orient='records')

dblp['predictions'] = dblp['predictions'].apply(lambda x: float(x[0]))
pmc['predictions'] = pmc['predictions'].apply(lambda x: float(x[0]))
pmc = pmc[pmc['author_count']<100]


grouped = dblp.groupby('author_count')['predictions'].agg(['mean', 'std'])
coefficients = np.polyfit(grouped.index, grouped['mean'], deg=3)
polynomial = np.poly1d(coefficients)
x_smooth = np.linspace(grouped.index.min(), grouped.index.max(), 500)
y_smooth = polynomial(x_smooth)
y1 = y_smooth - polynomial(grouped.index).std()
y2 = y_smooth + polynomial(grouped.index).std()

plt.plot(x_smooth, y_smooth, color="#F27B35", label='Predictions')
plt.fill_between(x_smooth, y1, y2, color="#F27B35", alpha=0.5, label='Error Band')
plt.xlabel('Number of Authors')
plt.ylabel('Predictions')
plt.title('DBLP')
plt.legend(loc='lower right')
# plt.show()
plt.tight_layout()
pmc_mae_pdf_path = 'dblp_nature.pdf'
plt.savefig(pmc_mae_pdf_path)

import numpy as np
import matplotlib.pyplot as plt

grouped = pmc.groupby('author_count')['predictions'].agg(['mean', 'std'])

coefficients = np.polyfit(grouped.index, grouped['mean'], deg=3)
polynomial = np.poly1d(coefficients)
x_smooth = np.linspace(grouped.index.min(), grouped.index.max(), 500)
y_smooth = polynomial(x_smooth)
y1 = y_smooth - polynomial(grouped.index).std()
y2 = y_smooth + polynomial(grouped.index).std()
plt.plot(x_smooth, y_smooth, color='#457ABF', label='Predictions')
plt.fill_between(x_smooth, y1, y2, color='#457ABF', alpha=0.5, label='Error Band')

plt.xlabel('Number of Authors')
plt.ylabel('Predictions')
plt.title('PubMed')
plt.legend(loc='lower left')
# plt.legend()
# plt.show()

plt.tight_layout()
pmc_mae_pdf_path = 'pmc_nature.pdf'
plt.savefig(pmc_mae_pdf_path)