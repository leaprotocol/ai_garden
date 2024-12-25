import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const defaultOptions = {
  model: `minicpm-v`,//'minicpm-v',//'llava:7b',//'llama3.2-vision',
  prompt: `Rate these aspects 0-10:

Technical: (sharpness, exposure, noise)
0: unusable/corrupted
10: perfect technical execution

Composition: (balance, framing)
0: random/accidental
10: masterful composition

Artistic: (emotion, uniqueness)
0: no impact
10: museum-worthy

People: (expressions, poses)
0: awkward/unflattering
10: natural/flattering

Overall:
0: poor
10: exceptional

Output five numbers only, separated by commas.`,
  batchSize: 5,
  raw: false,
  host: 'localhost',
  headers: {
    'Authorization': 'Bearer 9a0923f6c0c2fe300a532a587b54659393f06b97fb58cc0c4883c6f67b911467'
  },
  temperature: 0.3
};

export function getOptions(targetDir = path.join(__dirname, 'photos')) {
  return {
    ...defaultOptions,
    targetDir
  };
} 