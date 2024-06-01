function createAudioHTML(path) {
    return '<audio controls controlslist="nodownload" class="px-1"> <source src=' +
        path +
        ' type="audio/wav">Your browser does not support the audio element.</audio>';
  }
  const numPerPage = 5;
  
  function generateContinuationTable(tableId, filenames, page) {
    let table = document.getElementById(tableId);
  
    let nrRows = table.rows.length;
    for (let i = 1; i < nrRows; i++) {
      table.deleteRow(1);
    }
    let prefix = 'audio_samples/prompt2/';
    for (let i = (page - 1) * numPerPage; i < page * numPerPage; i++) {
      let row = table.insertRow(i % numPerPage + 1);
      let cell = row.insertCell(0);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_original.flac');
  
      cell = row.insertCell(1);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_tts.wav');

      cell = row.insertCell(2);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_st2at.wav');
    }
  }

  function generateTransferTable(tableId, filenames, page) {
    let table = document.getElementById(tableId);
  
    let nrRows = table.rows.length;
    for (let i = 1; i < nrRows; i++) {
      table.deleteRow(1);
    }
    let prefix = 'audio_samples/prompt3/';
    for (let i = (page - 1) * numPerPage; i < page * numPerPage; i++) {
      let row = table.insertRow(i % numPerPage + 1);
      let cell = row.insertCell(0);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_original.flac');
  
      cell = row.insertCell(1);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_prompt.wav');

      cell = row.insertCell(2);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_st2at.wav');
    }
  }

  function generateUnconditionalTable(tableId, filenames, page) {
    let table = document.getElementById(tableId);
  
    let nrRows = table.rows.length;
    for (let i = 1; i < nrRows; i++) {
      table.deleteRow(1);
    }
    let prefix = 'audio_samples/prompt0/';
    for (let i = (page - 1) * numPerPage; i < page * numPerPage; i++) {
      let row = table.insertRow(i % numPerPage + 1);
      let cell = row.insertCell(0);
      cell.innerHTML = createAudioHTML(prefix + filenames[i] + '_pred.wav');
    }
  }

  

  const librispeechTestCleanContinuationFilenames = [
    '304',
    '1069',
    '749',
    '1223',
    '638',
  ];

  const librispeechTestCleanTransferFilenames = [
    '160',
    '202',
    '544',
    '605',
    '641',
    '880',
    '942',
    '990',
    '1058',
    '1201',
  ];

  const librispeechTestCleanUnconditionalFilenames = [
    '60',
    '173',
    '184',
    '283',
    '419',
    '527',
    '542',
    '852',
    '854',
    '1201',
  ];
  
  generateContinuationTable(
      'semantic-to-acoustic-table', librispeechTestCleanContinuationFilenames,
      1);
  generateTransferTable(
      'speaker-identity-transfer-table', librispeechTestCleanTransferFilenames,
      1);
  generateUnconditionalTable(
      'unconditional-generation-table', librispeechTestCleanUnconditionalFilenames,
      1);

  $(document).ready(function() {
    for (let i = 1; i <= 1; i++) {
      let id = '#semantic-to-acoustic-page-' + i;
      $(id).click(function() {
        generateContinuationTable(
            'semantic-to-acoustic-table',
            librispeechTestCleanContinuationFilenames, i);
        $(id).parent().siblings().removeClass('active');
        $(id).parent().addClass('active');
        return false;
      });
    }

    for (let i = 1; i <= 2; i++) {
      let id = '#speaker-identity-transfer-page-' + i;
      $(id).click(function() {
        generateTransferTable(
            'speaker-identity-transfer-table',
            librispeechTestCleanTransferFilenames, i);
        $(id).parent().siblings().removeClass('active');
        $(id).parent().addClass('active');
        return false;
      });
    }

    for (let i = 1; i <= 2; i++) {
      let id = '#unconditional-generation-page-' + i;
      $(id).click(function() {
        generateUnconditionalTable(
            'unconditional-generation-table',
            librispeechTestCleanUnconditionalFilenames, i);
        $(id).parent().siblings().removeClass('active');
        $(id).parent().addClass('active');
        return false;
      });
    }
  
  });