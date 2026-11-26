// Normalise author-name variants across the whole Zotero library.
// Zotero 7: Tools -> Developer -> Run JavaScript, tick "Run as async function", paste, Run.
// First run with DRY_RUN = true and read the list; then set DRY_RUN = false and run again.

const DRY_RUN = true;

// [current lastName, current firstName, fieldMode (0 = two fields, 1 = single field), new lastName, new firstName]
const RULES = [
  ['Garfield', 'E.', 0, 'Garfield', 'Eugene'],
  ['Garfield, Eugene', '', 0, 'Garfield', 'Eugene'],

  ['Van Eck', 'Nees Jan', 0, 'van Eck', 'Nees Jan'],
  ['Van Eck', 'N. J.', 0, 'van Eck', 'Nees Jan'],
  ['van Eck', 'N. J.', 0, 'van Eck', 'Nees Jan'],
  ['van Eck', 'N.', 0, 'van Eck', 'Nees Jan'],
  ['Eck', 'Nees Jan van', 0, 'van Eck', 'Nees Jan'],
  ['Nees Jan van Eck', '', 0, 'van Eck', 'Nees Jan'],
  ['Nees Jan van Eck', '', 1, 'van Eck', 'Nees Jan'],

  ['Waltman', 'L.', 0, 'Waltman', 'Ludo'],

  ['van Leeuwen', 'Thed', 0, 'van Leeuwen', 'Thed N.'],
  ['van Leeuwen', 'T. N.', 0, 'van Leeuwen', 'Thed N.'],
  ['Van Leeuwen', 'T. N.', 0, 'van Leeuwen', 'Thed N.'],
  ['Van Leeuwen', 'Th. N.', 0, 'van Leeuwen', 'Thed N.'],
  // NOT 'S. Van Leeuwen' (2009 doctoral-student paper): a different person.

  ['Van Raan', 'Anthony F. J.', 0, 'van Raan', 'Anthony F. J.'],
  ['Van Raan', 'Anthony FJ', 0, 'van Raan', 'Anthony F. J.'],
  ['Van Raan', 'A. F. J.', 0, 'van Raan', 'Anthony F. J.'],
  ['van Raan', 'A. F. J.', 0, 'van Raan', 'Anthony F. J.'],

  ['Tijssen', 'R.', 0, 'Tijssen', 'Robert J. W.'],
  ['Tijssen', 'R. J. W.', 0, 'Tijssen', 'Robert J. W.'],
  ['Noyons', 'E. C. M.', 0, 'Noyons', 'Ed C. M.'],
  ['Noyons', 'Ed', 0, 'Noyons', 'Ed C. M.'],

  // unspaced initials (these print with the last initial missing in APA/biblatex)
  ['Newell', 'W.H.', 0, 'Newell', 'W. H.'],
  ['Manstead', 'Antony S.R.', 0, 'Manstead', 'Antony S. R.'],
  ['Powell', 'Justin J.W.', 0, 'Powell', 'Justin J. W.'],
  ['Wood', 'F.Q.', 0, 'Wood', 'F. Q.'],

  // single-field names that should be two fields
  ['Cole, S', '', 1, 'Cole', 'Stephen'],
  ['Cole, J', '', 1, 'Cole', 'Jonathan R.'],
  ['Docampo, D.', '', 1, 'Docampo', 'Domingo'],
  ['Cram, L.', '', 1, 'Cram', 'Lawrence'],
  ['Stern, Nicholas', '', 1, 'Stern', 'Nicholas'],
  ['Peruginelli, G.', '', 1, 'Peruginelli', 'G.'],
  ['Faro, S.', '', 1, 'Faro', 'S.'],
  ['Agnoloni, T.', '', 1, 'Agnoloni', 'T.'],
  ['Urbanc, M.', '', 1, 'Urbanc', 'M.'],
  ['Ochsner, M.', '', 1, 'Ochsner', 'M.'],
];

const items = await Zotero.Items.getAll(Zotero.Libraries.userLibraryID, false, false);
const log = [];
let nItems = 0;
for (const item of items) {
  if (!item.isRegularItem()) continue;
  const creators = item.getCreators();
  let changed = false;
  for (const c of creators) {
    const fm = c.fieldMode ? 1 : 0;
    const first = c.firstName || '';
    for (const [l, f, m, nl, nf] of RULES) {
      if (c.lastName === l && first === f && fm === m) {
        log.push(`${item.key}  ${String(item.getField('date')).slice(0, 4)}  ${item.getField('title').slice(0, 55)}\n      "${first}|${l}" -> "${nf}|${nl}"`);
        c.lastName = nl;
        c.firstName = nf;
        c.fieldMode = 0;
        changed = true;
        break;
      }
    }
  }
  if (changed) {
    nItems++;
    if (!DRY_RUN) {
      item.setCreators(creators);
      await item.saveTx();
    }
  }
}
return (DRY_RUN ? 'DRY RUN - nothing saved\n' : 'SAVED\n') + nItems + ' items\n\n' + log.join('\n');
