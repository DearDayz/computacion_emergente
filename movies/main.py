from pathlib import Path
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	return df


def basic_info(df: pd.DataFrame):
	print("\n=== HEAD (primeras filas) ===")
	print(df.head())

	print("\n=== SHAPE ===")
	print(df.shape)

	print("\n=== INFO ===")
	df.info()

	print("\n=== DESCRIBE (numérico) ===")
	print(df.describe(include=[np.number]).transpose())

	print("\n=== DESCRIBE (objetos) ===")
	print(df.describe(include=[object]).transpose())

	print("\n=== Valores nulos por columna ===")
	print(df.isnull().sum().sort_values(ascending=False).head(30))

	print("\n=== Duplicados ===")
	print(f"Duplicados: {df.duplicated().sum()}")


def ensure_dir(d: Path):
	if not d.exists():
		d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, path: Path, dpi=150):
	ensure_dir(path.parent)
	fig.tight_layout()
	fig.savefig(path, dpi=dpi)
	plt.close(fig)


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path):
	wanted = ['vote_count', 'vote_average', 'popularity']
	num_cols = [c for c in wanted if c in df.columns]
	if not num_cols:
		return

	# Histogramas agrupados
	fig, axes = plt.subplots(nrows=min(4, len(num_cols)), ncols=1, figsize=(8, 4 * min(4, len(num_cols))))
	if len(num_cols) == 1:
		axes = [axes]
	for ax, col in zip(axes, num_cols):
		sns.histplot(df[col].dropna(), kde=True, ax=ax)
		ax.set_title(f'Histograma: {col}')
	save_fig(fig, out_dir / 'numeric_histograms_top4.png')

def plot_top_categories(df: pd.DataFrame, out_dir: Path, top_n=10):
    wanted = ['origin_country', 'original_language']
    obj_cols = [c for c in wanted if c in df.columns]
    for col in obj_cols:
        counts = df[col].value_counts(dropna=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f'Top {top_n} categorías en {col}')
        ax.set_xlabel('Conteo')
        save_fig(fig, out_dir / f'top_categories_{col}.png')

def plot_correlation(df: pd.DataFrame, out_dir: Path):
	num = df.select_dtypes(include=[np.number])
	if num.shape[1] < 2:
		return
	corr = num.corr()
	fig, ax = plt.subplots(figsize=(10, 8))
	sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
	ax.set_title('Matriz de correlación (num)')
	save_fig(fig, out_dir / 'correlation_matrix.png')


def main():
	parser = argparse.ArgumentParser(description='EDA básico para 10k_Poplar_Tv_Shows.csv')
	parser.add_argument('--dataset', '-d', type=str, default=None, help='Ruta al CSV (opcional)')
	parser.add_argument('--out', '-o', type=str, default='plots', help='Directorio de salida para las figuras (relativo a movies/)')
	args = parser.parse_args()

	base_dir = Path(__file__).parent
	if args.dataset:
		csv_path = Path(args.dataset)
	else:
		csv_path = base_dir / 'datasets' / '10k_Poplar_Tv_Shows.csv'

	if not csv_path.exists():
		print(f"ERROR: no se encontró el archivo CSV en: {csv_path}")
		sys.exit(1)

	out_dir = base_dir / args.out
	ensure_dir(out_dir)

	print(f'Leyendo datos desde: {csv_path}')
	df = load_data(csv_path)

	# Información en consola
	basic_info(df)

	# Plots
	print('\nGenerando gráficas y guardándolas en:', out_dir)
	try:
		plot_numeric_histograms(df, out_dir)
	except Exception as e:
		print('Advertencia: fallo en numeric histograms:', e)

	try:
		plot_top_categories(df, out_dir)
	except Exception as e:
		print('Advertencia: fallo en top categories:', e)

	try:
		plot_correlation(df, out_dir)
	except Exception as e:
		print('Advertencia: fallo en correlation plot:', e)

	print('\nEDA básico completado. Revisa el directorio de salida con las imágenes.')


if __name__ == '__main__':
	main()


