import random

import pygame

from nubrain.experiment_common.io import ExperimentIO
from nubrain.experiment_text_comprehension.wrap_text import draw_text_wrapped


def run_attention_task(
    *,
    io: ExperimentIO,
    screen: pygame.surface.Surface,
    screen_width: int,
    screen_height: int,
    background_color: tuple | list,
    stimulus_font: pygame.font.Font,
    stimulus_font_color: tuple | list,
    stimulus_class: str,
    stimulus_type: str,
    object_classes: list,
) -> dict:
    """
    Run one attention-probe trial.

    Asks which object class was shown on the just-completed trial, collects the response
    together with its reaction time, shows feedback, and (on an incorrect response) the
    correct answer.

    Returns a dict with keys:
        "log":            attention_task_log (question, answer options, responses)
        "was_correct":    True / False, or None if the participant quit before
                          answering
        "quit_requested": True if the experimenter/participant quit during the
                          task
    """
    if stimulus_type == "image":
        question_text = "Which was the last image?"
    else:
        question_text = "Which was the last word?"

    # Build answer options: the correct class plus up to three distractors. `is_match`
    # here means "this option equals the class shown this trial".
    options = [
        {"answer": object_class, "correct": (object_class == stimulus_class)}
        for object_class in object_classes
    ]
    correct_options = [opt for opt in options if opt["correct"]]
    incorrect_options = [opt for opt in options if not opt["correct"]]
    num_distractors = min(3, len(incorrect_options))
    answers = correct_options + random.sample(incorrect_options, num_distractors)
    random.shuffle(answers)

    # Draw the question and the answer options.
    screen.fill(background_color)
    y_pos = int(screen_height * 0.2)
    y_pos = draw_text_wrapped(
        surface=screen,
        text=question_text,
        font=stimulus_font,
        color=stimulus_font_color,
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
            font=stimulus_font,
            color=stimulus_font_color,
            y_start=y_pos,
            max_width=screen_width * 0.8,
            screen_width=screen_width,
        )
        y_pos += 30  # Spacing between answers

    pygame.display.flip()

    # Flush any lingering key presses from the previous question's feedback period (or
    # accidental double-taps).
    pygame.event.clear()

    # Capture start time (for reaction time), then drain the board buffer while we wait
    # for input.
    start_ticks = pygame.time.get_ticks()
    io.drain_eeg()

    # Wait for the participant's response.
    response_log = []
    response_correct = None  # stays None if the participant quits without answering
    quit_requested = False
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
                        response_time = (pygame.time.get_ticks() - start_ticks) / 1000.0
                        response_log.append(
                            {
                                "selected_answer_idx": selected_idx,
                                "is_correct": response_correct,
                                "response_time": response_time,
                            }
                        )
                        answered = True

    attention_task_log = {
        "question": question_text,
        "answers": answers,
        "response_log": response_log,
    }

    # Quit before answering: return without showing feedback.
    if quit_requested:
        return {
            "log": attention_task_log,
            "was_correct": response_correct,
            "quit_requested": True,
        }

    # Feedback (whether the response was correct).
    if response_correct:
        feedback_text = "Correct"
        feedback_color = (0, 255, 0)  # Green
    else:
        feedback_text = "Incorrect"
        feedback_color = (255, 0, 0)  # Red

    screen.fill(background_color)
    feedback_surface = stimulus_font.render(feedback_text, True, feedback_color)
    feedback_rect = feedback_surface.get_rect(
        center=(screen_width // 2, screen_height // 2)
    )
    screen.blit(feedback_surface, feedback_rect)
    pygame.display.flip()
    pygame.time.delay(1000)  # Pause for participant to read the feedback.

    # On an incorrect response, show the correct answer.
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
                font=stimulus_font,
                color=stimulus_font_color,
                y_start=y_pos,
                max_width=screen_width * 0.8,
                screen_width=screen_width,
            )
            y_pos += 60
            y_pos = draw_text_wrapped(
                surface=screen,
                text=correct_answer_text,
                font=stimulus_font,
                color=stimulus_font_color,
                y_start=y_pos,
                max_width=screen_width * 0.8,
                screen_width=screen_width,
            )
            pygame.display.flip()
            pygame.time.delay(1000)

    # Drain the board buffer again after the feedback period.
    io.drain_eeg()

    return {
        "log": attention_task_log,
        "was_correct": response_correct,
        "quit_requested": False,
    }
