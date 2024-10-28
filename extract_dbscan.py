import fitz
import pandas as pd
import numpy as np
import hdbscan

header_threshold = 0.15
footer_threshold = 0.9

def remove_hf(pdf_path):
	document = fitz.open(pdf_path)
	n_pages = document.page_count
	if n_pages == 1:
		document.insert_file(pdf_path)
		coordinates = {'x0': [], 'y0': [], 'x1': [], 'y1': [], 'width': [], 'height': []}
	for page in document:
		blocks = page.get_text('blocks')
	for block in blocks:
		coordinates['x0'].append(block[0])
		coordinates['y0'].append(block[1])
		coordinates['x1'].append(block[2])
		coordinates['y1'].append(block[3])
		coordinates['width'].append(block[2] - block[0])
		coordinates['height'].append(block[3] - block[1])
		
	df = pd.DataFrame(coordinates)
	
	quantile = 0.15
	upper = np.floor(df['y0'].quantile(1 - header_threshold + quantile))
	lower = np.ceil(df['y1'].quantile(footer_threshold - quantile))
	x_min = np.floor(df['x0'].min())
	x_max = np.ceil(df['x1'].max())
	y_min = np.floor(df['y0'].min())
	y_max = np.ceil(df['y1'].max())
	
	# Frequenza header/footer
	hff = 0.8
	min_clust = min_cluster_size = int(np.floor(n_pages * hff))
	if min_clust < 2:
		min_clust = 2
		hdbscan = hdbscan.HDBSCAN(min_cluster_size = min_clust)
		df['clusters'] = hdbscan.fit_predict(df)
		df_group = df.groupby('clusters').agg(
		avg_y0=('y0','mean'), avg_y1=('y1','mean'),
		std_y0=('y0','std'), std_y1=('y1','std'),
		max_y0=('y0','max'), max_y1=('y1','max'),
		min_y0=('y0','min'), min_y1=('y1','min'),
		cluster_size=('clusters','count'), avg_x0=('x0', 'mean')).reset_index()
		
	df_group = df_group.sort_values(['avg_y0', 'avg_y1'], ascending=[True, True])
	
	std = 0
	footer = np.floor(df_group[(np.floor(df_group['std_y0']) == std) & (np.floor(df_group['std_y1']) == std) & (df_group['min_y0'] >= upper) & (df_group['cluster_size'] <= n_pages)]['min_y0'].min())
	header = np.ceil(df_group[(np.floor(df_group['std_y0']) == std) & (np.floor(df_group['std_y1']) == std) & (df_group['min_y1'] <= lower) & (df_group['cluster_size'] <= n_pages)]['min_y1'].max())
	
	if not pd.isnull(footer):
		y_max = footer
	if not pd.isnull(header):
		y_min = header
	
	# Coordinate blocco del corpo del documento
	return x_min, y_min, x_max, y_max