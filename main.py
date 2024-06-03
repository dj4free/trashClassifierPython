from taipy.gui import Gui

logo_path = "elements/WasteWise.png"
content = ""

index = """
<|text-center|
<|{logo_path}|image|>

<|{content}|file_selector|>
select an image from your file system
>
"""

app = Gui(page=index)

if __name__ == '__main__':
    # use_reloader enables automatic reloading
    app.run(use_reloader=True)
