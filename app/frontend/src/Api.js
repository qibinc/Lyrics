import axios from 'axios';
import { URL, WORD, SEQ2SEQ } from './config';

export function queryWord(query) {
    return axios
        .get(URL + WORD, {
            params: {
                line: query
            }
        })
        .then(response => {
            return response.data;
        })
}

export function querySeq2Seq(query) {
    return axios
        .get(URL + SEQ2SEQ, {
            params: {
                line: query.split('，').join(',')
            }
        })
        .then(response => {
            return response.data;
        })
}
