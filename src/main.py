import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.recommender import MovieRecommender
from src.logger import logger

console = Console()

def display_results(query: str, recommendations: list):
    """Muestra las recomendaciones en una tabla elegante."""
    if not recommendations:
        console.print(f"[bold red]No se encontraron recomendaciones para:[/bold red] {query}")
        return

    table = Table(title=f"Películas similares a: {query}", title_style="bold magenta")
    table.add_column("Ranking", justify="center", style="cyan", no_wrap=True)
    table.add_column("Título de la Película", style="green")

    for i, movie in enumerate(recommendations, 1):
        table.add_row(str(i), movie)

    console.print(table)

def main():
    console.print(Panel.fit(
        "🎬 [bold blue]Movie Recommender System[/bold blue] 🎬\n[italic]Nivel Senior ML Portfolio[/italic]",
        border_style="bright_blue"
    ))

    try:
        # 1. Inicialización
        loader = DataLoader()
        processor = TextPreprocessor()
        recommender = MovieRecommender()

        # 2. Pipeline de datos con feedback visual
        with console.status("[bold green]Cargando y procesando datos...") as status:
            movies_df = loader.load_movies()
            processed_df = processor.create_soup(movies_df)
            feature_matrix = processor.get_feature_matrix(processed_df)
            recommender.fit(processed_df, feature_matrix)
        
        console.print("[bold check]✓ Sistema listo para operar.[/]\n")

        # 3. Bucle de interacción
        while True:
            query = Prompt.ask("[bold yellow]Introduce el nombre de una película[/] (o 'salir')")
            
            if query.lower() in ['salir', 'exit', 'q']:
                console.print("[bold blue]¡Hasta luego![/]")
                break

            recommendations = recommender.get_recommendations(query)
            display_results(query, recommendations)

    except KeyboardInterrupt:
        console.print("\n[bold red]Proceso interrumpido por el usuario.[/]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error crítico en la aplicación: {e}")
        console.print(f"[bold red]Error:[/] {e}")

if __name__ == "__main__":
    main()