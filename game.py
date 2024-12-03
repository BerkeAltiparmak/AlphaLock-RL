import random

def generate_secret_code(length=4):
    """Generate a random secret code of letters."""
    return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(length))

def get_feedback(secret_code, guess):
    """Provide feedback for the guess."""
    feedback = []
    used_indices = set()

    # Check for correct letters in the correct position
    for i in range(len(secret_code)):
        if guess[i] == secret_code[i]:
            feedback.append("Green")  # Correct letter and position
            used_indices.add(i)

    # Check for correct letters in the wrong position
    for i in range(len(secret_code)):
        if guess[i] != secret_code[i] and guess[i] in secret_code:
            for j in range(len(secret_code)):
                if guess[i] == secret_code[j] and j not in used_indices:
                    feedback.append("Yellow")  # Correct letter, wrong position
                    used_indices.add(j)
                    break

    # Fill remaining feedback with "Gray" for incorrect letters
    feedback.extend(["Gray"] * (len(secret_code) - len(feedback)))

    return feedback

def play_alphalock(provided_code=None):
    """Main function to play AlphaLock."""
    print("Welcome to AlphaLock!")
    print("Your task is to guess the secret 4-letter code in 10 tries.")
    print("Feedback: Green = correct letter and position, Yellow = correct letter but wrong position, Gray = incorrect letter.")
    print("The feedback does not indicate the location of the correct or incorrect letters, only the total count of correct letters or positions.")

    # Use the provided secret code or generate a random one
    if provided_code:
        secret_code = provided_code.upper()
        if len(secret_code) != 4 or not secret_code.isalpha():
            raise ValueError("Provided secret code must be a 4-letter combination of letters.")
    else:
        secret_code = generate_secret_code()

    attempts = 10

    while attempts > 0:
        guess = input(f"Enter your 4-letter guess ({attempts} attempts remaining): ").upper()

        # Validate input
        if len(guess) != 4 or not guess.isalpha():
            print("Invalid input! Please enter a 4-letter combination of letters.")
            continue

        # Check guess
        feedback = get_feedback(secret_code, guess)
        print("Feedback:", feedback)

        if feedback.count("Green") == 4:
            print(f"Congratulations! You've cracked the code: {secret_code}")
            break

        attempts -= 1

    if attempts == 0:
        print(f"Out of attempts! The secret code was: {secret_code}")

# Run the game
if __name__ == "__main__":
    print("You can provide a secret code or let the game generate one for you.")
    user_provided_code = input("Enter a secret code (or press Enter to generate one randomly): ").strip()
    play_alphalock(provided_code=user_provided_code if user_provided_code else None)
