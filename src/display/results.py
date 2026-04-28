# extern
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# local
from ..schemas.models import MinimalSearchResults


def print_msr(
    console: Console, msr: MinimalSearchResults, query: str
) -> None:
    console.print()
    console.print(
        Rule(
            title=f"[bold cyan] Search Results for: '{query}'[/]",
            style="cyan",
        )
    )

    console.print()

    for i, source in enumerate(msr.retrieved_sources):
        content = Text()
        content.append("File     : ", style="bold magenta")
        content.append(f"{source.file_path}\n", style="green")
        content.append("Position : ", style="bold magenta")

        content.append(
            f"Chars {source.first_character_index} ➔ "
            f"{source.last_character_index}",
            style="yellow",
        )

        panel = Panel(
            content,
            title=f"[bold yellow]Rank {i + 1}[/]",
            title_align="left",
            border_style="blue",
        )
        console.print(panel)

    console.print()
    console.print(
        Rule(
            title=(
                f"[bold cyan]Total msr.retrieved_sources: "
                f"{len(msr.retrieved_sources)}[/]"
            ),
            style="cyan",
        )
    )
    console.print()
