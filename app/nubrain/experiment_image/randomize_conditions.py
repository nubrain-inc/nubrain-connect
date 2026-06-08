import random
from collections import Counter


def shuffle_with_repetitions(
    *,
    list_with_duplicates: list,
    repetitions: int = 0,
    minimize_runs: bool = True,
) -> list:
    """
    Shuffle a list to have exactly 'repetitions' number of consecutive duplicates.

    A repetition is counted as each time an element is immediately followed by the same
    element. For example, [1, 1, 2, 2, 2] has 3 repetitions.

    Args:
        list_with_duplicates: Input list containing duplicates
        repetitions: Target number of consecutive duplicate pairs (default 0)
        minimize_runs: If True, minimize occurrences of 3+ consecutive identical items

    Returns:
        A shuffled list with exactly 'repetitions' consecutive duplicate pairs

    Raises:
        ValueError: If the target number of repetitions is not achievable
    """
    freq = Counter(list_with_duplicates)
    n = len(list_with_duplicates)

    # Calculate bounds for valid repetitions.
    max_repetitions = sum(count - 1 for count in freq.values())
    min_repetitions = 0

    # Check validity.
    if repetitions < min_repetitions or repetitions > max_repetitions:
        raise ValueError(
            f"Cannot achieve {repetitions} repetitions. "
            f"Valid range is {min_repetitions} to {max_repetitions}."
        )

    # Special case: max repetitions - group all same elements.
    if repetitions == max_repetitions:
        result = []
        values = list(freq.keys())
        random.shuffle(values)
        for value in values:
            result.extend([value] * freq[value])
        return result

    # Special case: zero repetitions - need perfect interleaving.
    if repetitions == 0:
        result = []
        remaining = dict(freq)
        last_val = None

        for _ in range(n):
            choices = [v for v in remaining.keys() if v != last_val]
            if not choices:
                choices = list(remaining.keys())

            choices.sort(key=lambda x: remaining[x], reverse=True)
            if len(choices) > 1 and remaining[choices[0]] == remaining[choices[1]]:
                next_val = random.choice(choices[:2])
            else:
                next_val = choices[0]

            result.append(next_val)
            remaining[next_val] -= 1
            if remaining[next_val] == 0:
                del remaining[next_val]
            last_val = next_val

        return result

    # General case with run minimization.
    max_attempts = 10000
    best_result = None
    best_run_count = float("inf")

    for attempt in range(max_attempts):
        result = []
        remaining = dict(freq)
        current_reps = 0
        consecutive_count = 0

        while len(result) < n:
            if not result:
                # First element - choose randomly.
                choices = list(remaining.keys())
                next_val = random.choice(choices)
                consecutive_count = 1
            else:
                last_val = result[-1]
                reps_needed = repetitions - current_reps
                items_left = n - len(result)

                # Can we repeat the last value?
                can_repeat_last = remaining.get(last_val, 0) > 0

                # Get other choices.
                other_choices = [v for v in remaining.keys() if v != last_val]

                # Calculate if we MUST repeat to meet target.
                max_future_reps = 0
                for v, cnt in remaining.items():
                    if v == last_val:
                        max_future_reps += cnt - 1
                    else:
                        max_future_reps += cnt - 1 if cnt > 0 else 0

                must_repeat = (
                    reps_needed > 0
                    and can_repeat_last
                    and max_future_reps < reps_needed
                )

                # Decide next value.
                if must_repeat:
                    # We must repeat to meet target.
                    next_val = last_val
                    current_reps += 1
                    consecutive_count += 1

                elif reps_needed > 0 and can_repeat_last and consecutive_count < 2:
                    # We need reps and haven't created a run yet. Calculate probability
                    # of repeating.
                    if minimize_runs:
                        # Higher probability to create pairs, lower for runs.
                        base_prob = min(0.6, reps_needed / max(items_left * 0.5, 1))
                        # Reduce probability if we already have one repetition.
                        if consecutive_count == 1:
                            prob = base_prob
                        else:
                            # Much lower prob for 3rd consecutive.
                            prob = base_prob * 0.3
                    else:
                        prob = min(0.7, reps_needed / max(items_left * 0.5, 1))

                    if random.random() < prob:
                        next_val = last_val
                        current_reps += 1
                        consecutive_count += 1
                    elif other_choices:
                        next_val = random.choice(other_choices)
                        consecutive_count = 1
                    else:
                        next_val = last_val
                        current_reps += 1
                        consecutive_count += 1

                elif reps_needed > 0 and can_repeat_last and consecutive_count >= 2:
                    # We've already created a pair, strongly prefer switching.
                    if minimize_runs and other_choices:
                        # Only continue run if absolutely necessary.
                        if max_future_reps < reps_needed - 1:
                            # Must continue the run.
                            next_val = last_val
                            current_reps += 1
                            consecutive_count += 1
                        else:
                            # Switch to different value.
                            next_val = random.choice(other_choices)
                            consecutive_count = 1
                    elif other_choices:
                        # Small chance to continue run.
                        if random.random() < 0.1:
                            next_val = last_val
                            current_reps += 1
                            consecutive_count += 1
                        else:
                            next_val = random.choice(other_choices)
                            consecutive_count = 1
                    else:
                        next_val = last_val
                        current_reps += 1
                        consecutive_count += 1

                elif reps_needed == 0:
                    # Have enough repetitions, avoid creating more.
                    if other_choices:
                        next_val = random.choice(other_choices)
                        consecutive_count = 1
                    elif can_repeat_last:
                        next_val = last_val
                        current_reps += 1
                        consecutive_count += 1
                    else:
                        break
                else:
                    # Need reps but can't repeat last value.
                    if other_choices:
                        next_val = random.choice(other_choices)
                        consecutive_count = 1
                    else:
                        break

            result.append(next_val)
            remaining[next_val] -= 1
            if remaining[next_val] == 0:
                del remaining[next_val]

        # Verify result.
        actual_reps = sum(
            1 for i in range(1, len(result)) if result[i] == result[i - 1]
        )

        if actual_reps == repetitions:
            if minimize_runs:
                # Count runs of 3+ identical items.
                run_count = 0
                i = 0
                while i < len(result):
                    j = i
                    while j < len(result) and result[j] == result[i]:
                        j += 1
                    if j - i >= 3:
                        run_count += 1
                    i = j

                # Keep track of best result with minimum runs.
                if run_count < best_run_count:
                    best_result = result
                    best_run_count = run_count

                # If we found a solution with no runs (or minimal runs), return it.
                if run_count == 0 or attempt > max_attempts // 2:
                    return best_result if best_result else result
            else:
                return result

    # Return the best result found.
    if best_result is not None:
        return best_result

    # If we couldn't find a solution, raise an error.
    raise RuntimeError(
        f"Could not generate a sequence with exactly {repetitions} repetitions "
        f"after {max_attempts} attempts. Try running the function again."
    )


def count_runs(lst: list, min_length: int = 3) -> int:
    """Count the number of runs of min_length or more consecutive identical items."""
    if not lst:
        return 0

    runs = 0
    i = 0
    while i < len(lst):
        j = i
        while j < len(lst) and lst[j] == lst[i]:
            j += 1
        if j - i >= min_length:
            runs += 1
        i = j
    return runs


def create_balanced_list(*, image_categories: list, target_length: int):
    """
    Creates a list with approximately equal instances of each item.

    Args:
        image_categories: List of items to distribute
        target_length: Desired length of the output list

    Returns:
        List with approximately equal distribution of input items
    """
    if not image_categories or target_length <= 0:
        return []

    n = len(image_categories)
    base_count = target_length // n  # Minimum count for each item
    remainder = target_length % n  # Extra items to distribute

    result = []
    for idx, item in enumerate(image_categories):
        # First 'remainder' items get one extra copy.
        count = base_count + (1 if idx < remainder else 0)
        result.extend([item] * count)

    return result


def sample_next_image(
    *,
    next_image_category: str,
    category_to_filepath: dict,
    previous_image_file_path: str | None,
):
    while True:
        next_image_file_path = random.choice(category_to_filepath[next_image_category])
        if previous_image_file_path == next_image_file_path:
            print("Sampled same image twice in a row, will try again")
        else:
            break
    return next_image_file_path


def sample_with_min_distance(
    *,
    n_samples: int,
    lower: int,
    upper: int,
    min_distance: int = 1,
):
    """
    Sample `n_samples` unique integers from [lower, upper] such that every pair of
    sampled values (when sorted) is at least `min_distance` apart.

    Uses reduction; sample from a compressed range without constraints, then expand the
    values to enforce the minimum gap.

    Parameters
    ----------
    n_samples : int
        Number of integers to sample.
    lower : int
        Lower bound of the interval (inclusive).
    upper : int
        Upper bound of the interval (inclusive).
    min_distance : int
        Minimum distance between any two consecutive samples (sorted order).

    Returns
    -------
    list[int]
        A list of `n_samples` sampled integers satisfying the distance constraint,
        returned in sorted order.
    """
    if n_samples <= 0:
        return []

    range_size = upper - lower + 1
    required_size = n_samples + (n_samples - 1) * (min_distance - 1)

    if required_size > range_size:
        raise ValueError(
            f"Cannot sample {n_samples} values from [{lower}, {upper}] with "
            f"min_distance={min_distance} (need at least {required_size} "
            f"integers in the range, but only {range_size} available)."
        )

    # Sample from a compressed range where no gap constraint is needed.
    reduced_upper = upper - (n_samples - 1) * (min_distance - 1)
    samples = sorted(random.sample(range(lower, reduced_upper + 1), n_samples))

    # Expand: shift each successive value to restore the minimum gap.
    result = [val + i * (min_distance - 1) for i, val in enumerate(samples)]

    return result
