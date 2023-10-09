import PyInquirer
import regex
from pprint import pprint

style = PyInquirer.style_from_dict({
    PyInquirer.Token.Separator: '#cc5454',
    PyInquirer.Token.QuestionMark: '#673ab7 bold',
    PyInquirer.Token.Selected: '#cc5454',  # default
    PyInquirer.Token.Pointer: '#673ab7 bold',
    PyInquirer.Token.Instruction: '',  # default
    PyInquirer.Token.Answer: '#f44336 bold',
    PyInquirer.Token.Question: '',
})

class PhoneNumberValidator(PyInquirer.Validator):
    def validate(self, document):
        ok = regex.match('^([01]{1})?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\s?((?:#|ext\.?\s?|x\.?\s?){1}(?:\d+)?)?$', document.text)
        if not ok:
            raise PyInquirer.ValidationError(
                message='Please enter a valid phone number',
                cursor_position=len(document.text))  # Move cursor to end


class NumberValidator(PyInquirer.Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise PyInquirer.ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


questions = [
    {
        'type': 'checkbox',
        'message': 'Select toppings',
        'name': 'toppings',
        'choices': [
            PyInquirer.Separator('= The Meats ='),
            {
                'name': 'Ham'
            },
            {
                'name': 'Ground Meat'
            },
            {
                'name': 'Bacon'
            },
            PyInquirer.Separator('= The Cheeses ='),
            {
                'name': 'Mozzarella',
                'checked': True
            },
            {
                'name': 'Cheddar'
            },
            {
                'name': 'Parmesan'
            },
            PyInquirer.Separator('= The usual ='),
            {
                'name': 'Mushroom'
            },
            {
                'name': 'Tomato'
            },
            {
                'name': 'Pepperoni'
            },
            PyInquirer.Separator('= The extras ='),
            {
                'name': 'Pineapple'
            },
            {
                'name': 'Olives',
                'disabled': 'out of stock'
            },
            {
                'name': 'Extra cheese'
            }
        ],
        'validate': lambda answer: 'You must choose at least one topping.' \
            if len(answer) == 0 else True
    },
    {
        'type': 'confirm',
        'name': 'toBeDelivered',
        'message': 'Is this for delivery?',
        'default': True
    },
    {
        'type': 'input',
        'name': 'phone',
        'message': 'What\'s your phone number?'
    },
    {
        'type': 'list',
        'name': 'size',
        'message': 'What size do you need?',
        'choices': ['Large', 'Medium', 'Small'],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'input',
        'name': 'quantity',
        'message': 'How many do you need?',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'expand',
        'name': 'toppings',
        'message': 'What about the toppings?',
        'choices': [
            {
                'key': 'p',
                'name': 'Pepperoni and cheese',
                'value': 'PepperoniCheese'
            },
            {
                'key': 'a',
                'name': 'All dressed',
                'value': 'alldressed'
            },
            {
                'key': 'w',
                'name': 'Hawaiian',
                'value': 'hawaiian'
            }
        ]
    },
    {
        'type': 'rawlist',
        'name': 'beverage',
        'message': 'You also get a free 2L beverage',
        'choices': ['Pepsi', '7up', 'Coke']
    },
    {
        'type': 'input',
        'name': 'comments',
        'message': 'Any comments on your purchase experience?',
        'default': 'Nope, all good!'
    },
    {
        'type': 'list',
        'name': 'prize',
        'message': 'For leaving a comment, you get a freebie',
        'choices': ['cake', 'fries'],
        'when': lambda answers: answers['comments'] != 'Nope, all good!'
    }
]

answers = PyInquirer.prompt(questions, style=style)

pprint(answers)