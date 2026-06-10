import pygame

from nubrain.experiment_common.io import ExperimentIO
from nubrain.experiment_text_comprehension.wrap_text import draw_text_wrapped


def run_comprehension_questions(
    *,
    io: ExperimentIO,
    screen: pygame.surface.Surface,
    screen_width: int,
    screen_height: int,
    background_color: tuple | list,
    text_color: tuple | list,
    questions_and_answers: list,
    feedback_duration_ms: int = 1500,
    correct_answer_duration_ms: int = 3000,
) -> dict:
    """
    Run end-of-run multiple-choice comprehension questions.

    Presents each question with its (pre-defined) answer options, collects the
    response together with its reaction time, shows feedback, and on an
    incorrect response the correct answer.

    Returns a dict with keys:
        "n_correct":      number of questions answered correctly
        "response_log":   list of per-response dicts
        "quit_requested": True if the participant/experimenter quit during the
                          questions

    These questions appear after all EEG stimuli, so there is no EEG-locked
    timing here; reaction time uses the pygame millisecond clock as in the
    original. The board buffer is drained once per question so it does not fill
    during a long Q&A period.
    """
    qa_font = pygame.font.SysFont("arial", 42)
    feedback_font = pygame.font.SysFont("arial", 60, bold=True)

    n_correct = 0
    response_log = []
    quit_requested = False

    for q_idx, q_data in enumerate(questions_and_answers):
        if quit_requested:
            break

        question_text = q_data["question"]
        answers = q_data["answers"]

        # --- Draw the question and answer options ---
        screen.fill(background_color)
        y_pos = int(screen_height * 0.2)
        y_pos = draw_text_wrapped(
            surface=screen,
            text=question_text,
            font=qa_font,
            color=text_color,
            y_start=y_pos,
            max_width=screen_width * 0.8,
            screen_width=screen_width,
        )
        y_pos += 60  # Add extra spacing before options

        for a_idx, ans_data in enumerate(answers):
            ans_text = f"[{a_idx + 1}] {ans_data['answer']}"
            y_pos = draw_text_wrapped(
                surface=screen,
                text=ans_text,
                font=qa_font,
                color=text_color,
                y_start=y_pos,
                max_width=screen_width * 0.8,
                screen_width=screen_width,
            )
            y_pos += 30  # Spacing between answers

        pygame.display.flip()

        # Flush any lingering key presses from the previous question's feedback
        # period (or accidental double-taps).
        pygame.event.clear()

        # Capture start time (for reaction time).
        start_ticks = pygame.time.get_ticks()

        # --- Wait for the participant's response ---
        response_correct = None  # stays None if the participant quits
        answered = False
        while not answered and not quit_requested:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_requested = True
                    # Map keys 1-9 (number row or numpad) to answer indices 0-8.
                    elif (
                        pygame.K_1 <= event.key <= pygame.K_9
                        or pygame.K_KP1 <= event.key <= pygame.K_KP9
                    ):
                        if pygame.K_1 <= event.key <= pygame.K_9:
                            selected_idx = event.key - pygame.K_1
                        else:
                            selected_idx = event.key - pygame.K_KP1

                        # Only accept keys that map to a shown option.
                        if selected_idx < len(answers):
                            response_correct = answers[selected_idx]["correct"]
                            response_time = (
                                pygame.time.get_ticks() - start_ticks
                            ) / 1000.0
                            response_log.append(
                                {
                                    "question_idx": q_idx,
                                    "selected_answer_idx": selected_idx,
                                    "is_correct": response_correct,
                                    "response_time": response_time,
                                }
                            )
                            answered = True

        # Quit during this question: stop without showing feedback (matches the
        # original behaviour of breaking out at this point).
        if quit_requested:
            break

        # Keep the board buffer drained during the (potentially long) Q&A period.
        io.drain_eeg()

        # --- Feedback (whether the response was correct) ---
        if response_correct:
            n_correct += 1
            feedback_text = "Correct"
            feedback_color = (0, 255, 0)  # Green
        else:
            feedback_text = "Incorrect"
            feedback_color = (255, 0, 0)  # Red

        screen.fill(background_color)
        feedback_surface = feedback_font.render(feedback_text, True, feedback_color)
        feedback_rect = feedback_surface.get_rect(
            center=(screen_width // 2, screen_height // 2)
        )
        screen.blit(feedback_surface, feedback_rect)
        pygame.display.flip()
        pygame.time.delay(feedback_duration_ms)  # Pause to read the feedback.

        # --- On an incorrect response, show the correct answer ---
        if not response_correct:
            correct_answer_text = None
            for a_idx, ans_data in enumerate(answers):
                if ans_data["correct"]:
                    correct_answer_text = f"[{a_idx + 1}] {ans_data['answer']}"
                    break

            if correct_answer_text is not None:
                screen.fill(background_color)
                y_pos = int(screen_height * 0.4)
                y_pos = draw_text_wrapped(
                    surface=screen,
                    text="The correct answer was:",
                    font=qa_font,
                    color=text_color,
                    y_start=y_pos,
                    max_width=screen_width * 0.8,
                    screen_width=screen_width,
                )
                y_pos += 60
                y_pos = draw_text_wrapped(
                    surface=screen,
                    text=correct_answer_text,
                    font=qa_font,
                    color=text_color,
                    y_start=y_pos,
                    max_width=screen_width * 0.8,
                    screen_width=screen_width,
                )
                pygame.display.flip()
                pygame.time.delay(correct_answer_duration_ms)

    return {
        "n_correct": n_correct,
        "response_log": response_log,
        "quit_requested": quit_requested,
    }
