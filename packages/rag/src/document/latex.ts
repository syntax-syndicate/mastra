import { RecursiveCharacterTransformer } from './character';
import { Language } from './types';

export class LatexTransformer extends RecursiveCharacterTransformer {
  constructor(
    options: {
      chunkSize?: number;
      chunkOverlap?: number;
      lengthFunction?: (text: string) => number;
      keepSeparator?: boolean | 'start' | 'end';
      addStartIndex?: boolean;
      stripWhitespace?: boolean;
    } = {},
  ) {
    const separators = RecursiveCharacterTransformer.getSeparatorsForLanguage(Language.LATEX);
    super({ separators, isSeparatorRegex: true, options });
  }
}