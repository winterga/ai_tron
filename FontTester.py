import pygame
import os

# Initialize Pygame
pygame.init()

# Screen setup
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Font Examples")
clock = pygame.time.Clock()

# List of fonts to display
fonts = ['arial', 'arialblack', 'bahnschrift', 'calibri', 'cambria', 'cambriamath', 'candara', 'comicsansms', 'consolas', 'constantia', 'corbel', 'couriernew', 'ebrima', 'franklingothicmedium', 'gabriola', 'gadugi', 'georgia', 'impact', 'inkfree', 'javanesetext', 'leelawadeeui', 'leelawadeeuisemilight', 'lucidaconsole', 'lucidasans', 'malgungothic', 'malgungothicsemilight', 'microsofthimalaya', 'microsoftjhenghei', 'microsoftjhengheiui', 'microsoftnewtailue', 'microsoftphagspa', 'microsoftsansserif', 'microsofttaile', 'microsoftyahei', 'microsoftyaheiui', 'microsoftyibaiti', 'mingliuextb', 'pmingliuextb', 'mingliuhkscsextb', 'mongolianbaiti', 'msgothic', 'msuigothic', 'mspgothic', 'mvboli', 'myanmartext', 'nirmalaui', 'nirmalauisemilight', 'palatinolinotype', 'sansserifcollection', 'segoefluenticons', 'segoemdl2assets', 'segoeprint', 'segoescript', 'segoeui', 'segoeuiblack', 'segoeuiemoji', 'segoeuihistoric', 'segoeuisemibold', 'segoeuisemilight', 'segoeuisymbol', 'segoeuivariable', 'simsun', 'nsimsun', 'simsunextb', 'sitkatext', 'sylfaen', 'symbol', 'tahoma', 'timesnewroman', 'trebuchetms', 'verdana', 'webdings', 'wingdings', 'yugothic', 'yugothicuisemibold', 'yugothicui', 'yugothicmedium', 'yugothicuiregular', 'yugothicregular', 'yugothicuisemilight', 'holomdl2assets', 'agencyfb', 'algerian', 'bookantiqua', 'arialrounded', 'baskervilleoldface', 'bauhaus93', 'bell', 'bernardcondensed', 'bodoni', 'bodoniblack', 'bodonicondensed', 'bodonipostercompressed', 'bookmanoldstyle', 'bradleyhanditc', 'britannic', 'berlinsansfb', 'berlinsansfbdemi', 'broadway', 'brushscript', 'bookshelfsymbol7', 'californianfb', 'calisto', 'castellar', 'centuryschoolbook', 'centaur', 'century', 'chiller', 'colonna', 'cooperblack', 'copperplategothic', 'curlz', 'dubai', 'dubaimedium', 'dubairegular', 'elephant', 'engravers', 'erasitc', 'erasdemiitc', 'erasmediumitc', 'felixtitling', 'forte', 'franklingothicbook', 'franklingothicdemi', 'franklingothicdemicond', 'franklingothicheavy', 'franklingothicmediumcond', 'freestylescript', 'frenchscript', 'footlight', 'garamond', 'gigi', 'gillsans', 'gillsanscondensed', 'gillsansultracondensed', 'gillsansultra', 'gloucesterextracondensed', 'gillsansextcondensed', 'centurygothic', 'goudyoldstyle', 'goudystout', 'harlowsolid', 'harrington', 'haettenschweiler', 'hightowertext', 'imprintshadow', 'informalroman', 'blackadderitc', 'edwardianscriptitc', 'kristenitc', 'jokerman', 'juiceitc', 'kunstlerscript', 'widelatin', 'lucidabright', 'lucidacalligraphy', 'lucidafaxregular', 'lucidafax', 'lucidahandwriting', 'lucidasansregular', 'lucidasansroman', 'lucidasanstypewriterregular', 'lucidasanstypewriter', 'lucidasanstypewriteroblique', 'magneto', 'maiandragd', 'maturascriptcapitals', 'mistral', 'modernno20', 'monotypecorsiva', 'extra', 'niagaraengraved', 'niagarasolid', 'ocraextended', 'oldenglishtext', 'onyx', 'msoutlook', 'palacescript', 'papyrus', 'parchment', 'perpetua', 'perpetuatitling', 'playbill', 'poorrichard', 'pristina', 'rage', 'ravie', 'msreferencesansserif', 'msreferencespecialty', 'rockwellcondensed', 'rockwell', 'rockwellextra', 'script', 'showcardgothic', 'snapitc', 'stencil', 'twcen', 'twcencondensed', 'twcencondensedextra', 'tempussansitc', 'vinerhanditc', 'vivaldi', 'vladimirscript', 'wingdings2', 'wingdings3', 'interfxh', 'interfxhregular', 'myriadcad', 'hyswlongfangsong', 'swastro', 'olfsimplesansocregular', 'swcomp', 'swgothe', 'swgothg', 'swgothi', 'swgrekc', 'swgreks', 'swisop1', 'swisop2', 'swisop3', 'swisot1', 'swisot2', 'swisot3', 'swital', 'switalc', 'switalt', 'swmap', 'swmath', 'swmeteo', 'swmono', 'swmusic', 'swromnc', 'swromnd', 'swromns', 'swromnt', 'swscrpc', 'swscrps', 'swsimp', 'swtxt', 'swgdt', 'swlink', 'adobearabicbold', 'adobearabicbolditalic', 'adobearabicitalic', 'adobearabicregular', 'adobefanheitistdbold', 'adobegothicstdbold', 'adobehebrewbold', 'adobehebrewbolditalic', 'adobehebrewitalic', 'adobehebrewregular', 'adobeheitistdregular', 'adobemingstdlight', 'adobemyungjostdmedium', 'adobepistd', 'adobesongstdlight', 'adobethaibold', 'adobethaibolditalic', 'adobethaiitalic', 'adobethairegular', 'courierstd', 'courierstdbold', 'courierstdboldoblique', 'courierstdoblique', 'kozgopr6nmedium', 'kozminpr6nregular', 'minionproregular', 'myriadproregular', 'simsunextg', 'codicon', 'elusiveiconswebfont', 'fontawesome47webfont', 'fontawesome5brandswebfont', 'fontawesome5regularwebfont', 'fontawesome5solidwebfont', 'materialdesignicons5webfont', 'materialdesignicons6webfont', 'phosphor', 'remixicon']

# Directory to save rendered font images
os.makedirs("font_samples", exist_ok=True)

# Render text for each font
background_color = (255, 255, 255)  # White
text_color = (0, 0, 0)  # Black

for font_name in fonts:
    try:
        # Load font
        font = pygame.font.SysFont(font_name, 48)
        # Render text
        text_surface = font.render(font_name, True, text_color)
        # Display on screen
        screen.fill(background_color)
        screen.blit(text_surface, (50, 50))
        pygame.display.flip()
        # Save as image
        pygame.image.save(screen, f"font_samples/{font_name}.png")
        # Wait for a moment to visualize
        pygame.time.wait(1000)
    except Exception as e:
        print(f"Could not render font {font_name}: {e}")

# Quit Pygame
pygame.quit()
