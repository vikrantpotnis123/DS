#!/usr/bin/python3.6
def test_prompt(prompt, retries=4, error_handling="Please try again"):
    while True:
        res = input(prompt)
        if res in ("Yes", "Yeah", "y", "Y", "yes", "go"):
            print("Your response is",res)
            return True
        if res in ("No", "Nah", "n", "N", "no", "no go"):
            print("Your response is",res)
            return False
        retries -= 1
        if (retries < 0):
            raise ValueError("Invalid user response")
        print(error_handling)

test_prompt("Go or no go?", 2)
