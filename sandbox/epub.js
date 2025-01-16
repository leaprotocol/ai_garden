import EPub from 'epub'
import * as fs from "node:fs";

const epub = new EPub("./books_for_analysis/Breath_ The New Science of a Lost Art by James Nestor.epub");


epub.on("end", function() {
  const chapters = [];

  const processChapter = (index, parent) => {
    if (index >= epub.flow.length) {
      fs.writeFileSync("book_structure.json", JSON.stringify(chapters, null, 2));
      console.log("Chapters saved to book_structure.json");
      return;
    }

    const chapter = epub.flow[index];
    epub.getChapter(chapter.id, function(error, text) {
      if (error) {
        console.error("Error fetching chapter:", error);
      } else {
        const chapterContent = {
          title: chapter.title,
          id: chapter.id,
          content: text,
          subchapters: [],
        };

        if (parent) {
          parent.subchapters.push(chapterContent);
          processChapter(index + 1, parent);  // Continue with the same parent for siblings
        } else {
          chapters.push(chapterContent);
          processChapter(index + 1, chapterContent);  // Pass the current chapter as parent for subchapters
        }
      }
    });
  };

  processChapter(0, null);
});

epub.parse();