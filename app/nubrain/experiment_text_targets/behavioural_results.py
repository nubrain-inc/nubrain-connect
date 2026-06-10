import pygame


def show_behavioural_results(
    screen: pygame.surface.Surface,
    screen_width: int,
    screen_height: int,
    background_color: tuple | list,
    text_color: tuple | list,
    n_hits: int,
    n_misses: int,
    n_false_alarms: int,
    display_duration_ms: int = 5000,
) -> None:
    """
    Display end-of-run hit / miss / false-alarm summary screen.

    Pure display; runs after all EEG stimuli, so there is no EEG-locked timing here.
    """
    screen.fill(background_color)

    title_font = pygame.font.Font(None, 72)
    title_surface = title_font.render("Experiment Complete", True, text_color)
    title_rect = title_surface.get_rect(
        center=(screen_width // 2, screen_height // 2 - 150)
    )
    screen.blit(title_surface, title_rect)

    results_font = pygame.font.Font(None, 56)
    result_lines = [
        (f"Hits: {n_hits}", -20),
        (f"Misses: {n_misses}", 40),
        (f"False Alarms: {n_false_alarms}", 100),
    ]
    for line_text, y_offset in result_lines:
        line_surface = results_font.render(line_text, True, text_color)
        line_rect = line_surface.get_rect(
            center=(screen_width // 2, screen_height // 2 + y_offset)
        )
        screen.blit(line_surface, line_rect)

    pygame.display.flip()
    pygame.time.wait(display_duration_ms)  # Show results for a few seconds.
