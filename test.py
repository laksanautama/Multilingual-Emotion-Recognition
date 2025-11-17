string = """jawaban: ya
alasan: teks tersebut menggambarkan kejadian yang tidak terduga, yaitu seseorang yang tiba-tiba ingin memeluk, yang memicu reaksi menghindar ('makelid') dan perasaan jengkel/marah ('jengah gati tiang') dari pembicara. reaksi menghindar dan perasaan jengkel ini merupakan respons terhadap sesuatu yang mengejutkan atau tidak diinginkan.
Input: Suud ngajum tiang lantas nagih ngelut. Tiang makelid. Ada ane sedek nginem sake nagih ngelut. Jengah gati tiang.
"""

def convert_output_text(output_text: str):
    """Convert the output text from the LLM to standardized 'yes' or 'no'."""
    output_text = output_text.strip().lower()
    if 'jawaban: ya' in output_text:
        return 'yes'
    elif 'jawaban: tidak' in output_text:
        return 'no'
    else:
            # Return a default or handle cases where neither 'ya' nor 'tidak' is found
        return 'no'
    
print(f"Converted output text: {convert_output_text(string)}")